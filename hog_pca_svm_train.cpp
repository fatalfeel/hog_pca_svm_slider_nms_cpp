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

typedef struct _TrainSet_t
{
    uint32  img_id;
    uint32  img_label;
}TrainSet_t;

typedef struct _CorpImage_t
{
    dlib::rectangle     drect;
    matrix<rgb_pixel>   img_rgb;
}CorpImage_t;

typedef struct _PcaSvm_t
{
    uint32      corp_id;
    int         nEigens;
    float       thres_hold;
    cv::PCA*    pca;
    Ptr<SVM>*   svm;
}PcaSvm_t;

static int                      s_thread_num    = 128;
static volatile int             s_signal_num    = 128; //must use volatile avoid release lock in cache
static volatile int             s_hog_done      = 0;
static volatile int             s_detect_done   = 0;
std::vector<matrix<rgb_pixel>>  s_pos_lst;
std::vector<matrix<rgb_pixel>>  s_neg_lst;
std::vector<cv::Mat>            s_trainfhogs_lst;
std::vector<CorpImage_t>        s_corpdata_lst;
std::vector<cv::Rect>           s_srcRects;
pthread_mutex_t                 s_main_lock;
////////////////////////////////////////////////////
pthread_mutex_t                 s_train_lock;
pthread_mutex_t                 s_planar_lock;
pthread_mutex_t                 s_hogdone_lock;
////////////////////////////////////////////////////
pthread_mutex_t                 s_img_lock;
pthread_mutex_t                 s_rect_lock;
pthread_mutex_t                 s_detectdone_lock;
////////////////////////////////////////////////////
pthread_cond_t                  s_hog_thread_cond;
pthread_cond_t                  s_corp_thread_cond;

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

///////////////////////////////////////////////////////////////////////
/////////////////////////////////Train/////////////////////////////////
///////////////////////////////////////////////////////////////////////
static void TrainData_Process(uint32_t img_id, int img_label, matrix<rgb_pixel>& img_rgb)
{
    pthread_mutex_lock(&s_train_lock);

    switch(img_label)
    {
        case  1:
            img_rgb = s_pos_lst[img_id];
            break;
        case -1:
            img_rgb = s_neg_lst[img_id];
            break;
    }

    pthread_mutex_unlock(&s_train_lock);
}

static void Planar_Process(cv::Mat cMat)
{
    pthread_mutex_lock(&s_planar_lock);

    s_trainfhogs_lst.push_back(cMat.clone());

    pthread_mutex_unlock(&s_planar_lock);
}

//static void HogDone_Process(uint32_t test_value)
static void HogDone_Process()
{
    pthread_mutex_lock(&s_hogdone_lock);

    //cout << "HogDone_Process: " << test_value << endl;

    s_hog_done++;

    if(s_hog_done >= s_signal_num)
        pthread_cond_signal(&s_hog_thread_cond);

    pthread_mutex_unlock(&s_hogdone_lock);
}

/*
there are 31 features in each cell of fhog
1.
array2d<matrix<float,31,1>> hog;
extract_fhog_features(img_rgb, hog);
cols = hog.nc(), rows = hog.nr()
there are cols*rows cells
there are 31 features in each cell(31FC)
2.
dlib::array<array2d<float>> planar_hog;
extract_fhog_features(img_rgb, planar_hog);
there are 31 different matrices, u=0~30
cols = planar_hog[u].nc(), rows = planar_hog[u].nr()
there are cols*rows features in each matrix
there are all first  of 31 features(31FC) in matrix planar_hog[0]
there are all second of 31 features(31FC) in matrix planar_hog[1]
...
*/
static void* Thread_ComputeHog(void* arg)
{
    TrainSet_t*                 train_one = (TrainSet_t*)arg;
    matrix<rgb_pixel>           img_rgb;

    //cout << "s1: " << train_one->corp_id << endl; //debug
    TrainData_Process(train_one->img_id, train_one->img_label, img_rgb);

    dlib::array<array2d<float>> planar_hog;
    extract_fhog_features(img_rgb, planar_hog);

    /*array2d<matrix<float,31,1>> hog; //debug
    extract_fhog_features(img_rgb, hog);
    cout << hog.nc() << endl;
    cout << hog.nr() << endl;
    cout << hog[0][0].nc() << endl;
    cout << hog[0][0].nr() << endl<< endl;
    cout << planar_hog.size() << endl;
    cout << planar_hog[0].nc() << endl;
    cout << planar_hog[0].nr() << endl << endl;*/

    for(uint32_t u=0; u<planar_hog.size(); u+=4) //we don't need all size() 31 features so k+=4
    {
        cv::Mat cMat= toMat(planar_hog[u]);
        Planar_Process(cMat);
    }

    HogDone_Process();

    return NULL;
}

static void compute_HOGs( int img_label, std::vector<matrix<rgb_pixel>>& img_lst, Size win_size )
{
    pthread_attr_t  pattr;
    pthread_t       pthread_id;

    pthread_attr_init(&pattr);
    pthread_attr_setdetachstate(&pattr, PTHREAD_CREATE_DETACHED);

    uint32_t    strider     = 0;
    uint32_t    remain_size = img_lst.size();
    while( remain_size > 0 )
    {
        if( remain_size >= s_thread_num )
            s_signal_num = s_thread_num;
        else
            s_signal_num = remain_size;

        std::vector<TrainSet_t*> trainset_lst;
        for(uint32_t z=0; z<s_signal_num; z++)
        {
            TrainSet_t* train_one   = (TrainSet_t*)malloc(sizeof(TrainSet_t));
            train_one->img_id       = strider*s_thread_num + z;
            train_one->img_label    = img_label;
            trainset_lst.push_back(train_one);

            pthread_create(&pthread_id, &pattr, Thread_ComputeHog, (void*)train_one);
        }
        pthread_mutex_lock(&s_main_lock);
        pthread_cond_wait(&s_hog_thread_cond, &s_main_lock);
        pthread_mutex_unlock(&s_main_lock);
        s_hog_done = 0;

        for(uint32_t i=0; i<trainset_lst.size(); i++)
        {
            TrainSet_t* train_one = trainset_lst[i];
            free(train_one);
        }
        trainset_lst.clear();

        strider++;
        remain_size -= s_signal_num;
    }

    pthread_attr_destroy(&pattr);
}

///////////////////////////////////////////////////////////////////////
/////////////////////////////detection/////////////////////////////////
///////////////////////////////////////////////////////////////////////
static void get_CorpImage(matrix<rgb_pixel> image, Size win_size)
{
    int                 windows_n_rows  = win_size.height;
    int                 windows_n_cols  = win_size.width;
    int                 StepSlide_row   = 32;
    int                 StepSlide_col   = 32;
    matrix<rgb_pixel>   clip_img;
    CorpImage_t         corp_data;

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
            corp_data.drect     = drect;
            corp_data.img_rgb   = clip_img;
            s_corpdata_lst.push_back(corp_data);
        }
    }
}

static void CorpData_Process(uint32_t corp_id, CorpImage_t& corp_image)
{
    pthread_mutex_lock(&s_img_lock);

    corp_image = s_corpdata_lst[corp_id];

    pthread_mutex_unlock(&s_img_lock);
}

static void Rect_Process(cv::Point p0, cv::Point p1)
{
    pthread_mutex_lock(&s_rect_lock);

    s_srcRects.emplace_back(p0, p1);

    pthread_mutex_unlock(&s_rect_lock);
}

//static void DetectDone_Process(uint32_t test_value)
static void DetectDone_Process()
{
    pthread_mutex_lock(&s_detectdone_lock);

    //cout << "DetectDone_Process: " << test_value << endl;

    s_detect_done++;

    if(s_detect_done >= s_signal_num)
        pthread_cond_signal(&s_corp_thread_cond);

    pthread_mutex_unlock(&s_detectdone_lock);
}

static void* Thread_CorpDetect(void* arg)
{
    PcaSvm_t*                   pcasvm = (PcaSvm_t*)arg;
    int                         feature_times;
    float                       predict_socre;
    cv::Point                   p0,p1;
    CorpImage_t                 corp_data;
    dlib::array<array2d<float>> planar_hog;
    cv::Mat                     predictMat(1, pcasvm->nEigens, CV_32FC1);

    //cout << "s2: " << arg << endl;

    //corp_data = s_corpdata_lst[*(uint32_t*)arg];  //slower because system do lock
    CorpData_Process(pcasvm->corp_id, corp_data);    //faster because user do lock
    extract_fhog_features(corp_data.img_rgb, planar_hog);

    feature_times   = 0;
    predict_socre   = 0.0f;
    for(uint32_t u=0; u<planar_hog.size(); u+=4) //we don't need all size() 31 features so k+=4
    {
        cv::Mat cMat= toMat(planar_hog[u]);
        cv::Mat dst = cMat.reshape(1, 1);

        (*pcasvm->pca).project(dst, predictMat);

        if( (*pcasvm->svm)->predict(predictMat) >= 1 )
        {
            predict_socre += 1.0f;
        }

        feature_times++;
    }

    if ( predict_socre / (float)feature_times >= pcasvm->thres_hold )
    {
        //cout << onehog.drect.left() << "," << onehog.drect.top() << "," << onehog.drect.right() << "," << onehog.drect.bottom() << endl;
        p0 = cv::Point(corp_data.drect.left(),   corp_data.drect.top());
        p1 = cv::Point(corp_data.drect.right(),  corp_data.drect.bottom());
        //srcRects.emplace_back(p0, p1);
        Rect_Process(p0, p1);
    }

    //if ( predict_socre > 1.0f ) //test for score
    //    cout << corp_data.drect.left() << "," << corp_data.drect.top() << "," << corp_data.drect.right() << "," << corp_data.drect.bottom() << ": " << predict_socre << endl;

    DetectDone_Process();

    return NULL;
}

static void detect_object(matrix<rgb_pixel> image, cv::PCA* pca, Ptr<SVM>* svm, int nEigens, Size win_size, float thres_hold = 0.4f)
{
    int                     feature_times;
    //float                 predict_socre;
    //float                 history_score;
    cv::Mat                 predictMat(1, nEigens, CV_32FC1);
    cv::Point               p0,p1;
    std::vector<cv::Rect>   dstRects;
    pthread_attr_t          pattr;
    pthread_t               pthread_id;

    pthread_attr_init(&pattr);
    pthread_attr_setdetachstate(&pattr, PTHREAD_CREATE_DETACHED);

    get_CorpImage(image, win_size);

    //history_score = 0.0f;

    uint32_t    strider     = 0;
    uint32_t    remain_size = s_corpdata_lst.size();
    while( remain_size > 0 )
    {
        if( remain_size >= s_thread_num )
            s_signal_num = s_thread_num;
        else
            s_signal_num = remain_size;

        /*must malloc memory pointer to pthread_create,
         *using one address send to pthread_create,
         *some of Thread_CorpDetect will get same corp_id*/
        std::vector<PcaSvm_t*> pca_svm_lst;
        for(uint32_t z=0; z<s_signal_num; z++)
        {
            PcaSvm_t* pcasvm    = (PcaSvm_t*)malloc(sizeof(PcaSvm_t));
            pcasvm->corp_id     = strider*s_thread_num + z;
            pcasvm->pca         = pca;
            pcasvm->svm         = svm;
            pcasvm->nEigens     = nEigens;
            pcasvm->thres_hold  = thres_hold;
            pca_svm_lst.push_back(pcasvm);

            pthread_create(&pthread_id, &pattr, Thread_CorpDetect, (void*)pcasvm);
        }
        pthread_mutex_lock(&s_main_lock);
        pthread_cond_wait(&s_corp_thread_cond, &s_main_lock);
        pthread_mutex_unlock(&s_main_lock);
        s_detect_done = 0;

        for(uint32_t i=0; i<pca_svm_lst.size(); i++)
        {
            PcaSvm_t* pcasvm = pca_svm_lst[i];
            free(pcasvm);
        }
        pca_svm_lst.clear();

        strider++;
        remain_size -= s_signal_num;
    }
    s_corpdata_lst.clear();

    nms(s_srcRects, dstRects, 0.2f, 0);
    for( int i=0; i<dstRects.size(); i++)
    {
        cv::Rect r = dstRects[i];
        dlib::rectangle drect(r.tl().x, r.tl().y, r.br().x, r.br().y);
        dlib::draw_rectangle(image, drect,rgb_pixel(255,0,0));
    }
    s_srcRects.clear();

    //test show
    cout << "close window to end" << endl << endl;      //test
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
    //std::vector<cv::Mat>          trainfhogs_lst;
    Size                            win_size = Size(128, 256);

    pthread_mutex_init(&s_main_lock,NULL);
    ////////////////////////////////////////////////////
    pthread_mutex_init(&s_train_lock,NULL);
    pthread_mutex_init(&s_planar_lock,NULL);
    pthread_mutex_init(&s_hogdone_lock,NULL);
    ////////////////////////////////////////////////////
    pthread_mutex_init(&s_img_lock, NULL);
    pthread_mutex_init(&s_rect_lock,NULL);
    pthread_mutex_init(&s_detectdone_lock,NULL);
    ////////////////////////////////////////////////////
    pthread_cond_init(&s_hog_thread_cond,NULL);
    pthread_cond_init(&s_corp_thread_cond,NULL);

////////////////////////hog////////////////////////
    cout << "hog calculate start" << endl;
    load_images("./pos", s_pos_lst, win_size);
    compute_HOGs( 1, s_pos_lst, win_size );
    positive_count = s_trainfhogs_lst.size();
    labels.assign( positive_count, +1 );

    load_images("./neg", s_neg_lst, win_size);
    compute_HOGs(-1, s_neg_lst, win_size );
    negative_count = s_trainfhogs_lst.size() - positive_count;
    labels.insert(labels.end(), negative_count, -1);

    //Load the train_images into a Matrix
    //cv::Mat desc_mat(s_trainfhogs_lst.size(), s_trainfhogs_lst[0].rows * s_trainfhogs_lst[0].cols, CV_8UC1); //pca only
    cv::Mat desc_mat(s_trainfhogs_lst.size(), s_trainfhogs_lst[0].rows * s_trainfhogs_lst[0].cols, CV_32FC1);
    for (uint32_t i=0; i<s_trainfhogs_lst.size(); i++)
    {
        //desc_mat.row(i) = train_images[i].reshape(1, 1) + 0;
        //s_trainfhogs_lst[i].copyTo(desc_mat.row(i));
        s_trainfhogs_lst[i].reshape(1, 1).copyTo(desc_mat.row(i)); //reshaped in compute_HOGs
    }

////////////////////////pca////////////////////////
    cout << "pca trained start" << endl;
    int nEigens     = s_trainfhogs_lst[0].rows * s_trainfhogs_lst[0].cols / 4; //downsample related u+=4
    cv::Mat average = cv::Mat();
    PCA pca_trainer(desc_mat, average, CV_PCA_DATA_AS_ROW, nEigens);
    cv::Mat data(desc_mat.rows, nEigens, CV_32FC1);
    //Project the train_images onto the PCA subspace
    for(uint32_t i=0; i<s_trainfhogs_lst.size(); i++)
    {
        cv::Mat projectedMat(1, nEigens, CV_32FC1);
        pca_trainer.project(desc_mat.row(i), projectedMat);
        //data.row(i) = projectedMat.row(0) + 0;
        //projectedMat.row(0).copyTo(data.row(i));
        projectedMat.copyTo(data.row(i));
    }
    //pca_save("pca_data.xml",pca_trainer);

    s_pos_lst.clear();
    s_neg_lst.clear();
    s_trainfhogs_lst.clear();

////////////////////////svm////////////////////////
    cout << "svm trained start" << endl;
    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setKernel( SVM::RBF );
    svm->setGamma(0.5);     //lower more boxes
    //svm->setCoef0(1.0);
    svm->setC(1.5);         //lower more boxes
    //svm->setNu(0.5);
    //svm->setP(1.0);
    svm->setTermCriteria( TermCriteria( CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 80000000, FLT_EPSILON ));
    svm->train( data, ROW_SAMPLE, labels );
    //svm->save("svm_data.xml");
    //svm->load("svm_data.xml");

//////////////////////detect test/////////////////////
    cout << "detect test start" << endl;
    std::vector<matrix<rgb_pixel>>  test_lst;
    load_images("./test", test_lst, win_size, false);
    uint64_t        frames = 0;
    double          elapsed,fps;
    struct timespec t1, t2;
    clock_gettime(CLOCK_REALTIME, &t1);

    //for (int k=0; k<100; k++) //for test fps
    {
        for(auto& img : test_lst)
        {
            detect_object(img, &pca_trainer, &svm, nEigens, win_size);
            frames++;
        }
    }
    test_lst.clear();

    clock_gettime(CLOCK_REALTIME, &t2);
    elapsed = ((t2.tv_sec - t1.tv_sec) * 1000000000 + t2.tv_nsec - t1.tv_nsec) / 1000000000.0;
    fps     = frames / elapsed;
    printf("detected frames=%lu\nelapsed time=%f\nfps=%f\n", frames, elapsed, fps);

//////////////////////destory/////////////////////
    pthread_cond_destroy(&s_corp_thread_cond);
    pthread_cond_destroy(&s_hog_thread_cond);
    ////////////////////////////////////////////////////
    pthread_mutex_destroy(&s_detectdone_lock);
    pthread_mutex_destroy(&s_rect_lock);
    pthread_mutex_destroy(&s_img_lock);
    ////////////////////////////////////////////////////
    pthread_mutex_destroy(&s_hogdone_lock);
    pthread_mutex_destroy(&s_planar_lock);
    pthread_mutex_destroy(&s_train_lock);
    ////////////////////////////////////////////////////
    pthread_mutex_destroy(&s_main_lock);

    //s_corpdata_lst.clear();   //in func detect_object
    //s_srcRects.clear();       //in func detect_object

    return EXIT_SUCCESS;
}

// ----------------------------------------------------------------------------------------

