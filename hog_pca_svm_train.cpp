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

static void load_images( const String& dirname, std::vector<matrix<rgb_pixel>>& img_lst, Size img_size, bool isTrain = true)
{
    std::vector<String> files;
    glob( dirname, files );

    for ( size_t i = 0; i < files.size(); ++i )
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
        for(uint32_t k=0; k<planar_hog.size(); k+=4) //we don't need all size() 31 features so k+=4
        {
            cv::Mat cMat = toMat(planar_hog[k]);
            gradient_lst.push_back(cMat.clone());
        }
    }
}

static void get_OneHOG(matrix<rgb_pixel>& img, std::vector<cv::Mat>& gradient_lst)
{
	dlib::array<dlib::array2d<float>> planar_hog;
	extract_fhog_features(img, planar_hog);
	for(uint32_t k=0; k<planar_hog.size(); k+=4) //we don't need all size() 31 features so k+=4
	{
		cv::Mat cMat = toMat(planar_hog[k]);
		gradient_lst.push_back(cMat.clone());
	}
}

//static void detect_object(matrix<rgb_pixel>& image, cv::PCA& pca, Ptr<SVM>& svm, int nEigens, Size win_size, float thres_hold = 0.01f)
static void detect_object(matrix<rgb_pixel> image, cv::PCA& pca, Ptr<SVM>& svm, int nEigens, Size win_size, float thres_hold = 0.01f) //no &image will keep original
{
    float                   feature_times;
    float                 	predict_socre;
    float                   history_score;
    cv::Mat             	predictMat(1, nEigens, CV_32FC1);
    std::vector<cv::Rect> 	srcRects;
    std::vector<cv::Rect> 	dstRects;
	cv::Point 				p0,p1;
    int                 	windows_n_rows  = win_size.height;
    int                 	windows_n_cols  = win_size.width;
    int                 	StepSlide_row   = 32;
    int                 	StepSlide_col   = 64;

    history_score = 0;

    for (int row = 0; row <= image.nr()- windows_n_rows; row += StepSlide_row)
    {
        for (int col = 0; col <= image.nc() - windows_n_cols; col += StepSlide_col)
        {
        	matrix<rgb_pixel>   	clip_img;
        	std::vector<cv::Mat>	detect_gradients;
        	dlib::rectangle rect(col, row, col+windows_n_cols-1, row+windows_n_rows-1);
            extract_image_chip(image, rect, clip_img);
            get_OneHOG(clip_img, detect_gradients);

            feature_times   = 0;
            predict_socre   = 0;
            for(uint32_t k=0; k<detect_gradients.size(); k+=4)
            {
                cv::Mat dst = detect_gradients[k].reshape(1, 1);
            	pca.project(dst, predictMat);

            	if( svm->predict(predictMat) >= 1 )
            	{
            	    predict_socre += 1.0f;
            	    history_score += 1.0f;
            	}

            	feature_times += 1.0f;
            }

            if ( predict_socre / feature_times >= thres_hold )
            {
                //draw_rectangle(image,rect,rgb_pixel(255,0,0));
                p0 = cv::Point(col, row);
                p1 = cv::Point(col+windows_n_cols-1, row+windows_n_rows-1);
                srcRects.emplace_back(p0, p1);
            }
        }
    }

    cout << history_score << endl; //test

    nms(srcRects, dstRects, 0.3f, 0);
    for( auto& r : dstRects)
    {
    	dlib::rectangle drect = dlib::rectangle(r.tl().x, r.tl().y, r.br().x, r.br().y);
    	draw_rectangle(image, drect,rgb_pixel(255,0,0));
    }

    //test
    /*image_window iwin;
    iwin.set_image(image);
    usleep(6000000);*/
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

	for (int k=0; k<200; k++) //for test loop
	{
		/*compute_HOGs( test_lst, test_gradients, win_size, false );
		for(auto& src : test_gradients)
		{
			cv::Mat dst = src.reshape(1, 1);
			cv::Mat predictMat(1, nEigens, CV_32FC1);
			pca_trainer.project(dst, predictMat);
			int ret = svm->predict(predictMat);
			cout << ret << endl;

			frames++;
		}*/
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

	return EXIT_SUCCESS;
}

// ----------------------------------------------------------------------------------------

