
#include "laserProcessingClass.h"
#include "orbextractor.h"
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <Eigen/Dense>
#include <unordered_set> // 添加這行頭文件  `

//多threads
#include <thread>
#include <vector>
#include <mutex>

#include "opencv2/img_hash.hpp"



void LaserProcessingClass::init(lidar::Lidar lidar_param_in){
    lidar_param = lidar_param_in;
}


void LaserProcessingClass::downSamplingToMap(const pcl::PointCloud<pcl::PointXYZI>::Ptr& surf_pc_in, pcl::PointCloud<pcl::PointXYZI>::Ptr& surf_pc_out){
    downSizeFilterSurf.setInputCloud(surf_pc_in);
    downSizeFilterSurf.filter(*surf_pc_out);
 
}

//surface===============================

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <opencv2/opencv.hpp>
#include <thread>
#include <vector>
#include <cmath>

void processPointCloudRegions_surface(const pcl::PointCloud<pcl::PointXYZI>::Ptr& surf_first, 
                                      int startIdx, int endIdx, 
                                      int half_window_size, double intensity_threshold, 
                                      double gradient_threshold, int window_size, 
                                      std::vector<pcl::PointXYZI>& planePixels) {
    for (int i = startIdx + half_window_size; i < endIdx - half_window_size; i = i + 4 ) {
        bool isPlanar = false;
        
        // 1. 检查 x 轴上的最大最小强度值是否在阈值内
        float minIntensity = std::numeric_limits<float>::max();
        float maxIntensity = std::numeric_limits<float>::min();

        for (int wx = -half_window_size; wx <= half_window_size; ++wx) {
            float x_value = std::abs(surf_first->points[i + wx].x); // 使用绝对值
            if (!std::isnan(x_value)) {
                minIntensity = std::min(minIntensity, x_value);
                maxIntensity = std::max(maxIntensity, x_value);
            }
        }

        if (std::abs(maxIntensity - minIntensity) <= intensity_threshold) { // 使用绝对值
            // 如果强度值在阈值内，则检查 y 轴上的梯度
            float depth_top = surf_first->points[i + half_window_size].z;
            float depth_center = surf_first->points[i].z;
            float depth_bottom = surf_first->points[i - half_window_size].z;

            double gradient_1 = std::abs(depth_top - depth_center);
            double gradient_2 = std::abs(depth_bottom - depth_center);

            double gradient = std::abs(gradient_1 - gradient_2);

            if (gradient <= gradient_threshold) {
                isPlanar = true;
            }
        }

        if (!isPlanar) {
            // 2. 检查 y 轴上的最大最小强度值是否在阈值内
            float minIntensity = std::numeric_limits<float>::max();
            float maxIntensity = std::numeric_limits<float>::min();

            for (int wy = -half_window_size; wy <= half_window_size; ++wy) {
                float y_value = std::abs(surf_first->points[i + wy].y); // 使用绝对值
                if (!std::isnan(y_value)) {
                    minIntensity = std::min(minIntensity, y_value);
                    maxIntensity = std::max(maxIntensity, y_value);
                }
            }

            if (std::abs(maxIntensity - minIntensity) <= intensity_threshold) { // 使用绝对值
                // 如果强度值在阈值内，则检查 x 轴上的梯度
                float depth_left = surf_first->points[i + half_window_size].z;
                float depth_center = surf_first->points[i].z;
                float depth_right = surf_first->points[i - half_window_size].z;

                double gradient_1 = std::abs(depth_left - depth_center);
                double gradient_2 = std::abs(depth_right - depth_center);

                double gradient = std::abs(gradient_1 - gradient_2);

                if (gradient <= gradient_threshold) {
                    isPlanar = true;
                }
            }
        }

        if (isPlanar) {
            planePixels.push_back(surf_first->points[i]);
        }
    }
}


void processPointCloud_surface(const pcl::PointCloud<pcl::PointXYZI>::Ptr& surf_first,
                               int half_window_size, double depth_threshold, 
                               double gradient_threshold, int window_size, 
                               std::vector<pcl::PointXYZI>& planePixels) {
    int numThreads = 8; // Number of threads to use
    std::vector<std::thread> threads;
    std::vector<std::vector<pcl::PointXYZI>> planePixelsList(numThreads);

    // Divide the point cloud into regions and assign threads
    int totalPoints = surf_first->points.size();
    int pointsPerThread = totalPoints / numThreads;
    int remainingPoints = totalPoints % numThreads;
    int startIdx = 0;

    for (int i = 0; i < numThreads; ++i) {
        int endIdx = startIdx + pointsPerThread;
        if (i == numThreads - 1) endIdx += remainingPoints;
        threads.emplace_back(processPointCloudRegions_surface, surf_first, startIdx, endIdx, 
                             half_window_size, depth_threshold, gradient_threshold, window_size, 
                             std::ref(planePixelsList[i]));
        startIdx = endIdx;
    }

    // Join threads
    for (auto& thread : threads) {
        thread.join();
    }

    // Combine results from each thread
    for (const auto& pixels : planePixelsList) {
        planePixels.insert(planePixels.end(), pixels.begin(), pixels.end());
    }
}


//=============edge=================================

void LaserProcessingClass::featureExtractionWithSobel(pcl::PointCloud<pcl::PointXYZI>::Ptr& pc_in,
                                                        pcl::PointCloud<pcl::PointXYZI>::Ptr& pc_out_edge,
                                                        pcl::PointCloud<pcl::PointXYZI>::Ptr& surf_first) {
    // 初始化Sobel算子
    std::vector<int> sobelX = {-1, 0, 1, -2, 0, 2, -1, 0, 1}; // 核心仅针对3x3部分
    std::vector<int> sobelY = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
    std::vector<int> sobelZ = {-1, -2, -1, 0, 0, 0, 1, 2, 1}; // 这需要调整，暂时沿用Y的形式

    int kernel_size = 3;

    // 遍历点云数据并计算Sobel梯度
    for (size_t i = kernel_size; i < pc_in->points.size() - kernel_size; i = i+4) {
        double grad_x = 0.0, grad_y = 0.0, grad_z = 0.0;

        for (int k = -1; k <= 1;k++) {
            int idx = i + k;
            grad_x += sobelX[(k + 1) * kernel_size + 1] * pc_in->points[idx].x; //使用x座標深淺來做梯度分析
            grad_y += sobelY[(k + 1) * kernel_size + 1] * pc_in->points[idx].y; //利用y座標深淺來做梯度分析
            grad_z += sobelZ[(k + 1) * kernel_size + 1] * pc_in->points[idx].z; //利用z座標深淺來做梯度分析
        }

        // 计算梯度幅值
        double grad_magnitude = sqrt(grad_x * grad_x + grad_y * grad_y + grad_z * grad_z);

        // 根据梯度幅值判断是边缘点还是平面点
        if (grad_magnitude > 0.5) {  // 根据需要调整阈值
            pc_out_edge->push_back(pc_in->points[i]);
        } else {
            surf_first->push_back(pc_in->points[i]);
        }
    }
}

//=========================================
///
void LaserProcessingClass::pointcloudtodepth(pcl::PointCloud<pcl::PointXYZI>::Ptr& pc_in,
                                             sensor_msgs::ImageConstPtr& image_msg, 
                                             Eigen::Matrix<double, 3, 4>& matrix_3Dto2D,
                                             Eigen::Matrix3d& result,
                                             Eigen::Matrix3d& RR,
                                             Eigen::Vector3d& tt,
                                             pcl::PointCloud<pcl::PointXYZI>::Ptr& surf_first,
                                             pcl::PointCloud<pcl::PointXYZI>::Ptr& pc_out_surf
                                             ) {

    // Processing 360-degree point cloud to extract planar features
    int window_size = 5;
    int half_window_size = window_size / 2;
    double depth_threshold = 0.6;
    double gradient_threshold = 0.5;

    std::vector<pcl::PointXYZI> planePixels;

    processPointCloud_surface(surf_first, half_window_size, depth_threshold, gradient_threshold, window_size, planePixels);
  
    pc_out_surf->points.insert(pc_out_surf->points.end(), planePixels.begin(), planePixels.end());
    
    std::cout << "after plane number = " << pc_out_surf->points.size() << std::endl;
}

void LaserProcessingClass::featureExtraction(pcl::PointCloud<pcl::PointXYZI>::Ptr& pc_in, 
                                             pcl::PointCloud<pcl::PointXYZI>::Ptr& pc_out_edge, 
                                             pcl::PointCloud<pcl::PointXYZI>::Ptr& surf_first,
                                             sensor_msgs::ImageConstPtr& image_msg, 
                                             Eigen::Matrix<double, 3, 4>& matrix_3Dto2D){

    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*pc_in, indices);
    // std::cout << "pcin number = " << (int)pc_in->points.size() << std::endl;

    int N_SCANS = lidar_param.num_lines;
    std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> laserCloudScans;
    for(int i=0;i<N_SCANS;i++){
        laserCloudScans.push_back(pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>()));
    }

    for (int i = 0; i < (int) pc_in->points.size(); i++)
    {
        // if(pc_in->points[i].x >= 0){
            int scanID=0;
            double distance = sqrt(pc_in->points[i].x * pc_in->points[i].x + pc_in->points[i].y * pc_in->points[i].y);
            if(distance<lidar_param.min_distance || distance>lidar_param.max_distance)
                continue;
            double angle = atan(pc_in->points[i].z / distance) * 180 / M_PI;
            
           
                if (angle >= -8.83)
                    scanID = int((2 - angle) * 3.0 + 0.5);
                else
                    scanID = N_SCANS / 2 + int((-8.83 - angle) * 2.0 + 0.5);

                if (angle > 2 || angle < -24.33 || scanID > 63 || scanID < 0)
                {
                    continue;
                }
            
            laserCloudScans[scanID]->push_back(pc_in->points[i]); 

    }

    
featureExtractionWithSobel(pc_in,pc_out_edge,surf_first);

    std::cout << "after edge number = " << (int)pc_out_edge->points.size() << std::endl;

}

LaserProcessingClass::LaserProcessingClass(){
    
}

Double2d::Double2d(int id_in, double value_in){
    id = id_in;
    value =value_in;
};

PointsInfo::PointsInfo(int layer_in, double time_in){
    layer = layer_in;
    time = time_in;
};