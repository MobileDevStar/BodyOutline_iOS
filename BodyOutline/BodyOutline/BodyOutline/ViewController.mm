//
//  ViewController.m
//  BodyOutline
//
//  Created by Simba on 7/8/19.
//  Copyright © 2019 Simba. All rights reserved.
//

#if defined __cplusplus
#import <opencv2/videoio/cap_ios.h>
#import <opencv2/core/core.hpp>
#import <opencv2/highgui/highgui.hpp>
#import <opencv2/imgproc/imgproc.hpp>
#import <opencv2/objdetect/objdetect.hpp>

using namespace cv;
using namespace std;
#endif

#define PI                        3.141592f
#define ROTATE_VIDEO              YES

NSString* face_cascade_name = [[NSBundle mainBundle] pathForResource:@"haarcascade_frontalface_alt2" ofType:@"xml"];

enum PROCSTATE {
    PROC_NONE,
    PROC_START,
    PROC_LOAD,
    PROC_RESHOW,
    PROC_END
};

//using namespace cv
#import "ViewController.h"

@interface ViewController () <CvVideoCameraDelegate>
{
    IBOutlet UIImageView *previewCamera;
    
    CvVideoCamera *videoCamera;
    
    PROCSTATE procState;
    
    cv::CascadeClassifier face_cascade;
    cv::CascadeClassifier eyes_cascade;
    
    HOGDescriptor hog;
    
    cv::Mat prevFrame;
    cv::Mat curFrame;
    cv::Mat nextFrame;
    
    float faceW;
    float faceH;
    cv::Point faceCenter;
    int frameCenterY;
}
@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view, typically from a nib.
    
    videoCamera = [[CvVideoCamera alloc] initWithParentView:previewCamera];
    videoCamera.defaultAVCaptureDevicePosition = AVCaptureDevicePositionFront;
    videoCamera.defaultAVCaptureSessionPreset = AVCaptureSessionPreset640x480;
    if (ROTATE_VIDEO) {
        videoCamera.defaultAVCaptureVideoOrientation = AVCaptureVideoOrientationLandscapeLeft;
        //videoCamera.recordVideo = YES;
        videoCamera.rotateVideo = YES;
    } else {
        videoCamera.defaultAVCaptureVideoOrientation = AVCaptureVideoOrientationPortrait;
    }
    videoCamera.defaultFPS = 30;
    videoCamera.grayscaleMode = NO;
    videoCamera.delegate = self;
    
    faceCenter = cv::Point(0,0);
    
    
    hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
    
    BOOL loadFace = face_cascade.load( [face_cascade_name UTF8String] );
    
    if (!loadFace) {
        [[[UIAlertView alloc] initWithTitle:NSLocalizedString(@"LOADING ERROR", @"")
                                    message:NSLocalizedString(@"Couldn't load xml files of opencv", @"")
                                   delegate:self
                          cancelButtonTitle:nil
                          otherButtonTitles:NSLocalizedString(@"OK", @""), nil] show];
    }
    
    procState = PROC_NONE;
}

- (void)viewDidAppear:(BOOL)animated
{
    [super viewDidAppear:animated];
    
    [videoCamera layoutPreviewLayer];
    [videoCamera start];
    
    [videoCamera saveVideo];
    
    
    
}

- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

#pragma mark - Protocol CvVideoCameraDelegate

#ifdef __cplusplus
- (void)processImage:(cv::Mat&)src;
{
    Mat image_copy;
    
    cvtColor(src, image_copy, CV_BGRA2GRAY);
    
    if (ROTATE_VIDEO) {
        //rotate mat 90 degree because camera video was rotated
   //     transpose(image_copy, image_copy);
    //    flip(image_copy, image_copy, 0);
    }
    
    Mat frame_gray;
    
    prevFrame = curFrame;
    curFrame = nextFrame;
    nextFrame = image_copy.clone();
    
    cv::Mat edgeMat = [self GetOutline:self->prevFrame curFrame:self->curFrame nextFrame:self->nextFrame];
    cvtColor(edgeMat, src, CV_GRAY2BGRA);
    
    equalizeHist(image_copy, image_copy);
    
    BOOL bFaceFound = false;
    vector<cv::Rect> faces;
    
    self->face_cascade.detectMultiScale(image_copy, faces, 1.1, 3, 0 | CV_HAAR_SCALE_IMAGE, cv::Size(30, 30));
    
    for(unsigned int i = 0; i < faces.size(); ++i) {
        cv::Point center = cv::Point( faces[0].x + faces[0].width*0.5, faces[0].y + faces[0].height*0.5 );
        faceW = faces[0].width;
        faceH = faces[0].height;
        faceCenter = center;

        // show face detection part
        ellipse( src, faceCenter, cv::Size( faceW*0.5, faceH*0.5), 0, 0, 360, cv::Scalar( 255, 0, 0 ), 4, 8, 0 );
        //            [self testShow:sframe];
        
        bFaceFound = true;
    }
    NSLog(@"detected: %d", bFaceFound);

    vector<cv::Rect> found, found_filtered;
    double t = (double)getTickCount();
    // run the detector with default parameters. to get a higher hit-rate
    // (and more false alarms, respectively), decrease the hitThreshold and
    // groupThreshold (set groupThreshold to 0 to turn off the grouping completely).
    hog.detectMultiScale(image_copy, found, 0, cv::Size(8, 8), cv::Size(32, 32), 1.05, 2, false);
    t = (double)getTickCount() - t;
    printf("tdetection time = %gms\n", t*1000. / cv::getTickFrequency());
    size_t i, j;
    for (i = 0; i < found.size(); i++)
    {
        cv::Rect r = found[i];
        for (j = 0; j < found.size(); j++)
            if (j != i && (r & found[j]) == r)
                break;
        if (j == found.size())
            found_filtered.push_back(r);
    }
    for (i = 0; i < found_filtered.size(); i++)
    {
        cv::Rect r = found_filtered[i];
        // the HOG detector returns slightly larger rectangles than the real objects.
        // so we slightly shrink the rectangles to get a nicer output.
        r.x += cvRound(r.width*0.1);
        r.width = cvRound(r.width*0.8);
        r.y += cvRound(r.height*0.07);
        r.height = cvRound(r.height*0.8);
        rectangle(src, r.tl(), r.br(), cv::Scalar(0, 255, 0), 3);
    }
}
#endif

-(cv::Mat) GetOutline:(cv::Mat)pFrame curFrame:(cv::Mat)cFrame nextFrame:(cv::Mat) nFrame
{
    cv::Mat threshImg;
    
    int highThresh = threshold(nFrame, threshImg, 70, 255, THRESH_BINARY + THRESH_OTSU);
    int lowThreshold = 0.5* highThresh;
    NSLog(@"highThresh: %d", highThresh);
    NSLog(@"lowThresh: %d", lowThreshold);
    //int highThreshold = thre
   // int lowThreshold = 70;
    int ratio = 3;
    int kernel_size = 3;
    
    cv::Mat detected_edges;
    cv::Mat new_detected_edges;
    
    if (!pFrame.empty() && !cFrame.empty() && !nFrame.empty()) {
        cv::Mat diffFrame1, diffFrame2, outFrame;
        absdiff(pFrame, cFrame, diffFrame1);
        absdiff(cFrame, nFrame, diffFrame2);
        bitwise_and(diffFrame1, diffFrame2, outFrame);
        
        blur( diffFrame1, detected_edges, cv::Size(3,3) );
        /// Canny detector
        //Canny( detected_edges, new_detected_edges, lowThreshold, lowThreshold*ratio, kernel_size );
        Canny( detected_edges, new_detected_edges, lowThreshold, highThresh, kernel_size );
        /// Using Canny¡¦s output as a mask, we display our result
    }
    return new_detected_edges;
}

//detect's the face in cv::Mat and displays rect around face.
- (BOOL) detectAndDisplay:( cv::Mat )frame
{
    BOOL bFaceFound = false;
    vector<cv::Rect> faces;
    Mat frame_gray;
    
//    rectangle(frame, cv::Point(100, 100),
//              cv::Point(50, 50),
//              cv::Scalar(0,255,255));
    
    cvtColor(frame, frame_gray, CV_BGRA2GRAY);
    
    
    equalizeHist(frame_gray, frame_gray);
    
   
    
    face_cascade.detectMultiScale(frame_gray, faces, 1.1, 3, 0 | CV_HAAR_SCALE_IMAGE, cv::Size(30, 30));
    
    for(unsigned int i = 0; i < faces.size(); ++i) {
        rectangle(frame, cv::Point(faces[i].x, faces[i].y),
                  cv::Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height),
                  cv::Scalar(0,255,255));
        bFaceFound = true;
    }
    
    
    
    NSLog(@"detected: %d", bFaceFound);
    return bFaceFound;
}

@end
