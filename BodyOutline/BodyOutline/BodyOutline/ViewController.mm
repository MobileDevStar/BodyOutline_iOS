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

using namespace cv;
#endif

#define PI                        3.141592f
#define ROTATE_VIDEO              YES

#import "ViewController.h"

@interface ViewController () <CvVideoCameraDelegate>
{
    IBOutlet UIImageView *previewCamera;
    
    CvVideoCamera *videoCamera;
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
}

- (void)viewDidAppear:(BOOL)animated
{
    [super viewDidAppear:animated];
    
    [videoCamera layoutPreviewLayer];
    [videoCamera start];
    
    [videoCamera saveVideo];
    
}
/*
#pragma mark - Protocol CvVideoCameraDelegate

#ifdef __cplusplus
- (void)processImage:(cv::Mat&)src;
{
    cv::Mat image = src.clone();
    
    if (ROTATE_VIDEO) {
        //rotate mat 90 degree because camera video was rotated
        transpose(image, image);
        flip(image, image, 0);
    }
    
    cv::Mat image_rotate;

}
#endif*/
@end
