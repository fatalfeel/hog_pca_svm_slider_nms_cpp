#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unistd.h>
#include <sys/time.h>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/ml.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#include "dlib/opencv.h"
#include "dlib/gui_widgets.h"
#include "dlib/image_io.h"
#include "dlib/image_transforms.h"
#include "nms.h"

using namespace std;
using namespace cv;
using namespace cv::ml;
using namespace dlib;

typedef struct _CorpImage_t
{
	dlib::rectangle		drect;
	matrix<rgb_pixel>   img_rgb;;
}CorpImage_t;

typedef struct _CorpHog_t
{
	dlib::rectangle 		drect;
	std::vector<cv::Mat>	feature_lst;
}CorpHog_t;

static int					s_thread_num = 32;
static volatile int			s_signal_num = 32; //must use volatile avoid release lock in cache
std::vector<CorpImage_t>	s_corpdata_lst;
std::vector<CorpHog_t>		s_hogdata_lst;
pthread_mutex_t 			s_main_lock;
pthread_mutex_t 			s_img_lock;
pthread_mutex_t 			s_hog_lock;
pthread_cond_t 				s_thread_cond;

static void load_images( const String& dirname, std::vector<matrix<rgb_pixel>>& img_lst, Size img_size, bool isTrain = true)
{
    std::vector<String> files;
    glob( dirname, files );

    for ( size_t i = 0; i < files.size(); i++ )
    {
        //pca only
        /*cv::Mat gray;
        cv::Mat img = imread( files[i] ); // load the image 0=IMREAD_GRAYSCALE

        if ( img.empty() )
        {
            cout << files[i] << " is invalid!" << endl; // invalid image, skip it.
            continue;
        }

        if( img.cols*img.rows != imgsize.width * imgsize.height )
            cv::resize( img, img, imgsize, 0, 0, INTER_LINEAR_EXACT);
        cvtColor(img, gray, COLOR_BGR2GRAY);
        img_lst.push_back( gray );*/

        //hog + pca
        matrix<rgb_pixel> img_rgb;
    	load_image(img_rgb, files[i]);

    	if( isTrain )
    	{
            if( img_rgb.nc() != img_size.width || img_rgb.nr() != img_size.height )
            {
                dlib::matrix<float> size_mask(img_size.height,img_size.width);
                dlib::resize_image(img_rgb, size_mask);
            }
    	}

    	img_lst.push_back( img_rgb );
    }
}

static void compute_HOGs( std::vector<matrix<rgb_pixel>>& img_lst, std::vector<cv::Mat>& gradient_lst, Size win_size, bool use_flip )
{
    for( size_t i = 0 ; i < img_lst.size(); i++ )
    {
        /*array2d<matrix<float,31,1>> hog;
        extract_fhog_features(img_lst[i], hog);
        float* ptrMat = (float*)image_data(hog);
        for(int k=0; k<31; k++)
        {
            cv::Mat cMat = cv::Mat(hog.nr(), hog.nc(), CV_32FC1, ptrMat);
            gradient_lst.push_back(cMat.clone());
            ptrMat += hog.nr() * hog.nc();
        }
        gradient_lst.push_back(cMat.clone());
        //cout << "ok" << endl;*/

        dlib::array<dlib::array2d<float>> planar_hog;
        extract_fhog_features(img_lst[i], planar_hog);
        for(uint32_t u=0; u<planar_hog.size(); u+=4) //we don't need all size() 31 features so k+=4
        {
            cv::Mat cMat = toMat(planar_hog[u]);
            gradient_lst.push_back(cMat.clone());
        }
    }
}

static void get_OneHOG(matrix<rgb_pixel>& img, std::vector<cv::Mat>& gradient_lst)
{
	dlib::array<dlib::array2d<float>> planar_hog;
	extract_fhog_features(img, planar_hog);
	for(uint32_t u=0; u<planar_hog.size(); u+=4) //we don't need all size() 31 features so k+=4
	{
		cv::Mat cMat = toMat(planar_hog[u]);
		gradient_lst.push_back(cMat.clone());
	}
}

static void Image_Process(int corp_pos, CorpImage_t& corp_image)
{
    pthread_mutex_lock(&s_img_lock);

    corp_image = s_corpdata_lst[corp_pos];

    pthread_mutex_unlock(&s_img_lock);
}

static void Hog_Process(CorpHog_t hog_data)
{
    pthread_mutex_lock(&s_hog_lock);

    s_hogdata_lst.push_back(hog_data);

    if(s_hogdata_lst.size() >= s_signal_num)
    	pthread_cond_signal(&s_thread_cond);

    pthread_mutex_unlock(&s_hog_lock);
}

static void* Thread_OneCorp(void* arg)
{
	CorpHog_t 							hog_data;
	CorpImage_t							corp_data;
	dlib::array<dlib::array2d<float>> 	planar_hog;

	Image_Process(*(uint32_t*)arg, corp_data);
	//cout << "s3 " << arg << endl;

	hog_data.drect = corp_data.drect;
	//cout << "s2:" << hog_data.drect.left() << "," << hog_data.drect.top() << "," << hog_data.drect.right() << "," << hog_data.drect.bottom() << endl;
	extract_fhog_features(corp_data.img_rgb, planar_hog);
	for(uint32_t u=0; u<planar_hog.size(); u+=4) //we don't need all size() 31 features so k+=4
	{
		cv::Mat cMat = toMat(planar_hog[u]);
		hog_data.feature_lst.push_back(cMat.clone());
	}

	Hog_Process(hog_data);

    return NULL;
}

//static void detect_object(matrix<rgb_pixel>& image, cv::PCA& pca, Ptr<SVM>& svm, int nEigens, Size win_size, float thres_hold = 0.01f)
static void detect_object(matrix<rgb_pixel> image, cv::PCA& pca, Ptr<SVM>& svm, int nEigens, Size win_size, float thres_hold = 0.01f) //no &image will keep original
{
    int                     feature_times;
    float                	predict_socre;
    float                 	history_score;
    cv::Mat             	predictMat(1, nEigens, CV_32FC1);
    cv::Point               p0,p1;
    std::vector<cv::Rect> 	srcRects;
    std::vector<cv::Rect> 	dstRects;
    int                 	windows_n_rows  = win_size.height;
    int                 	windows_n_cols  = win_size.width;
    int                 	StepSlide_row   = 32;
    int                 	StepSlide_col   = 64;
    pthread_attr_t          pattr;
    pthread_t               pthread_id;
    matrix<rgb_pixel>   		clip_img;
    std::vector<CorpImage_t>	corpdata_lst;
    CorpImage_t					corp_data;

    pthread_attr_init(&pattr);
    pthread_attr_setdetachstate(&pattr, PTHREAD_CREATE_DETACHED);

    history_score = 0.0f;

    for (int row = 0; row <= image.nr()- windows_n_rows; row += StepSlide_row)
    {
        for (int col = 0; col <= image.nc() - windows_n_cols; col += StepSlide_col)
        {
        	dlib::rectangle drect(col, row, col+windows_n_cols-1, row+windows_n_rows-1);
        	/*drect.set_left(col);
        	drect.set_top(row);
        	drect.set_right(col+windows_n_cols-1);
        	drect.set_bottom(row+windows_n_rows-1);*/

        	extract_image_chip(image, drect, clip_img);
        	corp_data.drect 	= drect;
        	corp_data.img_rgb 	= clip_img;
        	s_corpdata_lst.push_back(corp_data);
        }
    }

    uint32_t 	strider		= 0;
    uint32_t 	remain_size	= s_corpdata_lst.size();
    while( remain_size > 0 )
    {
    	if( remain_size >= s_thread_num )
    		s_signal_num = s_thread_num;
    	else
    		s_signal_num = remain_size;

    	/*must malloc memory pointer to pthread_create,
    	 *using one address send to pthread_create,
    	 *some of Thread_OneCorp will get same corp_pos*/
    	uint32_t* 				corp_pos;
    	std::vector<uint32_t*> 	corp_pos_lst;
    	for(uint32_t z=0; z<s_signal_num; z++)
		{
    		corp_pos 	= (uint32_t*)malloc(sizeof(uint32_t));
    		//cout << "s1 " << corp_pos << endl;
    		corp_pos_lst.push_back(corp_pos);
    		*corp_pos	= strider*s_thread_num + z;
    		pthread_create(&pthread_id, &pattr, Thread_OneCorp, (void*)corp_pos);
		}
    	pthread_mutex_lock(&s_main_lock);
		pthread_cond_wait(&s_thread_cond, &s_main_lock);
		pthread_mutex_unlock(&s_main_lock);

		for(uint32_t i=0; i<corp_pos_lst.size(); i++)
		{
			corp_pos = corp_pos_lst[i];
			//cout << "s2 " << corp_pos << endl;
			free(corp_pos);
		}
		corp_pos_lst.clear();

		for(uint32 i=0; i<s_hogdata_lst.size(); i++) //the most s_hogdata_lst is cpu threads number
		{
			CorpHog_t hog_data	= s_hogdata_lst[i];
			feature_times 		= 0;
			predict_socre 		= 0.0f;

			for(uint32_t u=0; u<hog_data.feature_lst.size(); u+=4) //the most feature_lst is 31
			{
				cv::Mat dst = hog_data.feature_lst[u].reshape(1, 1);
				pca.project(dst, predictMat);

				if( svm->predict(predictMat) >= 1 )
				{
					predict_socre += 1.0f;
					history_score += 1.0f;
				}

				feature_times++;
			}
			hog_data.feature_lst.clear(); //the most feature_lst is 31

			if ( predict_socre / (float)feature_times >= thres_hold )
			{
				//cout << onehog.drect.left() << "," << onehog.drect.top() << "," << onehog.drect.right() << "," << onehog.drect.bottom() << endl;
				p0 = cv::Point(hog_data.drect.left(),	hog_data.drect.top());
				p1 = cv::Point(hog_data.drect.right(),	hog_data.drect.bottom());
				srcRects.emplace_back(p0, p1);
			}
		}
		s_hogdata_lst.clear(); //the most s_hogdata_lst is cpu threads number

		strider++;
		remain_size -= s_signal_num;
    }
    s_corpdata_lst.clear();

    cout << "hscore: " << history_score << endl; //test

    nms(srcRects, dstRects, 0.3f, 0);
    for( int i=0; i<dstRects.size(); i++)
    {
    	cv::Rect r = dstRects[i];
    	dlib::rectangle drect(r.tl().x, r.tl().y, r.br().x, r.br().y);
    	dlib::draw_rectangle(image, drect,rgb_pixel(255,0,0));
    }

    //test show
    cout << "close window to end" << endl; 		//test
    image_window iwin;
    iwin.set_image(image);
    iwin.wait_until_closed();
    //usleep(2000000);

    pthread_attr_destroy(&pattr);
}

static void pca_load(const string &file_name, cv::PCA _pca)
{
    FileStorage fs(file_name,FileStorage::READ);
    fs["mean"]      >> _pca.mean ;
    fs["e_vectors"] >> _pca.eigenvectors ;
    fs["e_values"]  >> _pca.eigenvalues ;
    fs.release();
}

static void pca_save(const string &file_name, cv::PCA _pca)
{
    FileStorage fs(file_name,FileStorage::WRITE);
    fs << "mean" << _pca.mean;
    fs << "e_vectors" << _pca.eigenvectors;
    fs << "e_values" << _pca.eigenvalues;
    fs.release();
}

int main(int argc, char** argv)
{
    int                             positive_count;
    int                             negative_count;
    std::vector<int>                labels;
    std::vector<matrix<rgb_pixel>>  pos_lst, neg_lst, test_lst;
    std::vector<cv::Mat>            train_gradients, test_gradients;
    Size                            win_size = Size(128, 256);

	pthread_mutex_init(&s_main_lock,NULL);
	pthread_mutex_init(&s_img_lock, NULL);
    pthread_mutex_init(&s_hog_lock,NULL);
    pthread_cond_init(&s_thread_cond,NULL);

    load_images("./pos", pos_lst, win_size);
    compute_HOGs( pos_lst, train_gradients, win_size, false );
    positive_count = train_gradients.size();
    labels.assign( positive_count, +1 );

    load_images("./neg", neg_lst, win_size);
    compute_HOGs( neg_lst, train_gradients, win_size, false );
    negative_count = train_gradients.size() - positive_count;
    labels.insert(labels.end(), negative_count, -1);

    //Load the train_images into a Matrix
    //cv::Mat desc_mat(train_gradients.size(), train_gradients[0].rows * train_gradients[0].cols, CV_8UC1); //pca only
    cv::Mat desc_mat(train_gradients.size(), train_gradients[0].rows * train_gradients[0].cols, CV_32FC1);
    for (uint32_t i=0; i<train_gradients.size(); i++)
    {
        //desc_mat.row(i) = train_images[i].reshape(1, 1) + 0;
        //train_gradients[i].copyTo(desc_mat.row(i));
        train_gradients[i].reshape(1, 1).copyTo(desc_mat.row(i)); //reshaped in compute_HOGs
    }

    printf("pca trained start\n");
    int nEigens     = train_gradients[0].rows * train_gradients[0].cols / 4; //downsample related k+=4
    cv::Mat average = cv::Mat();
    PCA pca_trainer(desc_mat, average, CV_PCA_DATA_AS_ROW, nEigens);
    cv::Mat data(desc_mat.rows, nEigens, CV_32FC1);
    //Project the train_images onto the PCA subspace
    for(uint32_t i=0; i<train_gradients.size(); i++)
    {
        cv::Mat projectedMat(1, nEigens, CV_32FC1);
        pca_trainer.project(desc_mat.row(i), projectedMat);
        //data.row(i) = projectedMat.row(0) + 0;
        //projectedMat.row(0).copyTo(data.row(i));
        projectedMat.copyTo(data.row(i));
    }
    //pca_save("pca_data.xml",pca_trainer);

    printf("svm trained start\n");
    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setKernel( SVM::RBF );
    svm->setGamma(0.55); //lower more boxes
	//svm->setCoef0(1.0);
	svm->setC(1.5);     //lower more boxes
	//svm->setNu(0.5);
	//svm->setP(1.0);
	svm->setTermCriteria( TermCriteria( CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 1000000, FLT_EPSILON ));
    svm->train( data, ROW_SAMPLE, labels );
    //svm->save("svm_data.xml");
    //svm->load("svm_data.xml");

    printf("test start\n");
    load_images("./test", test_lst, win_size, false);
    uint64_t		frames = 0;
	double 			elapsed,fps;
	struct timespec t1, t2;
	clock_gettime(CLOCK_REALTIME, &t1);

	//for (int k=0; k<1000; k++) //for test fps
	{
		for(auto& img : test_lst)
        {
            detect_object(img, pca_trainer, svm, nEigens, win_size);
            frames++;
        }
	}

    clock_gettime(CLOCK_REALTIME, &t2);
	elapsed = ((t2.tv_sec - t1.tv_sec) * 1000000000 + t2.tv_nsec - t1.tv_nsec) / 1000000000.0;
	fps 	= frames / elapsed;
	printf("detected frames=%lu\nelapsed time=%f\nfps=%f\n", frames, elapsed, fps);

	pthread_cond_destroy(&s_thread_cond);
	pthread_mutex_destroy(&s_hog_lock);
	pthread_mutex_destroy(&s_img_lock);
	pthread_mutex_destroy(&s_main_lock);

	return EXIT_SUCCESS;
}

// ----------------------------------------------------------------------------------------

