#include "solve_pnp.h"

RM_SolveAngle::RM_SolveAngle(){
    //读取摄像头标定xml文件
    FileStorage fs(file_path, FileStorage::READ);

    //读取相机内参和畸变矩阵
    fs["camera-matrix"] >> cameraMatrix;
    fs["distortion"] >> distCoeffs;
    cout << cameraMatrix << endl;
    cout << distCoeffs << endl;
    cout << "RM_SolveAngle is readied" << endl;
}


void RM_SolveAngle::run_SolvePnp(RotatedRect &rect, float _W, float _H){

    float half_x = _W * 0.5;
    float half_y = _H * 0.5;

    object_3d.clear();
    object_3d.push_back(Point3f(-half_x, -half_y, 0));
    object_3d.push_back(Point3f(half_x, -half_y, 0));
    object_3d.push_back(Point3f(half_x, half_y, 0));
    object_3d.push_back(Point3f(-half_x, half_y, 0));

    vertex_Sort(rect);
    
    solvePnP(object_3d, target2d, cameraMatrix, distCoeffs, rvec, tvec, false, SOLVEPNP_ITERATIVE);

    Mat ptz = camera_ptz(tvec);
    //cout << ptz << "-----" << rect.center << endl;

    get_Angle(ptz);
}

void RM_SolveAngle::vertex_Sort(RotatedRect & box){
    Point2f vertex[4];
    Point2f lu, ld, ru, rd;

    box.points(vertex);

    sort(vertex, vertex + 4, [](const Point2f & p1, const Point2f & p2) { return p1.x < p2.x; });

    if (vertex[0].y < vertex[1].y){
        lu = vertex[0];
        ld = vertex[1];
    }
    else{
        lu = vertex[1];
        ld = vertex[0];
    }
    if (vertex[2].y < vertex[3].y)	{
        ru = vertex[2];
        rd = vertex[3];
    }
    else {
        ru = vertex[3];
        rd = vertex[2];
    }

    target2d.clear();
    target2d.push_back(lu);
    target2d.push_back(ru);
    target2d.push_back(rd);
    target2d.push_back(ld );
}

Mat RM_SolveAngle::camera_ptz(Mat & t){
    //设相机坐标系绕X轴你是逆时针旋转θ后与云台坐标系的各个轴向平行
    double theta = 0;/*-atan(static_cast<double>(ptz_camera_y + barrel_ptz_offset_y))/static_cast<double>(overlap_dist);*/
    double r_data[] = {1,0,0,0,cos(theta),sin(theta),0,-sin(theta),cos(theta)};
    //设相机坐标系的原点在云台坐标系中的坐标为(x0,y0,z0)
    double t_data[] = {static_cast<double>(ptz_camera_x),static_cast<double>(ptz_camera_y),static_cast<double>(ptz_camera_z)};
    Mat r_camera_ptz(3,3,CV_64FC1,r_data);
    Mat t_camera_ptz(3,1,CV_64FC1,t_data);

    Mat position_in_ptz = r_camera_ptz * t - t_camera_ptz;
    //cout << position_in_ptz << endl;
    return position_in_ptz;
}

void RM_SolveAngle::get_Angle(const Mat & pos_in_ptz){
    //计算子弹下坠补偿
    const double *_xyz = (const double *)pos_in_ptz.data;
    //cout << "x:" << _xyz[0] << "   y:" << _xyz[1] << "   z:" << _xyz[2] << endl;
    double bullet_speed = 1.0;
    double down_t = 0.0;
    if(bullet_speed > 10e-3){
        down_t = _xyz[2] / bullet_speed;
    }

    double offset_gravity = 0.5 * 9.8 * down_t*down_t;
    double xyz[3] = {_xyz[0], _xyz[1] /*- offset_gravity*/, _xyz[2]};

    //计算角度
    if(barrel_ptz_offset_y != 0.f)
    {
        double alpha = 0.0, Beta = 0.0;
        alpha = asin(static_cast<double>(barrel_ptz_offset_y)/sqrt(xyz[1]*xyz[1] + xyz[2]*xyz[2]));

        if(xyz[1] < 0)
        {
            Beta = atan(-xyz[1]/xyz[2]);
            angle_y = static_cast<float>(-(alpha+Beta)); //camera coordinate
        }else if(xyz[1] < static_cast<double>(barrel_ptz_offset_y))
        {
            Beta = atan(xyz[1]/xyz[2]);
            angle_y = static_cast<float>(-(alpha - Beta));
        }else
        {
            Beta = atan(xyz[1]/xyz[2]);
            angle_y = static_cast<float>((Beta-alpha));   // camera coordinate
        }
    }else
    {
        angle_y = static_cast<float>(atan2(xyz[1],xyz[2]));
    }
    if(barrel_ptz_offset_x != 0.f)
    {
        double alpha = 0.0, Beta = 0.0;
        alpha = asin(static_cast<double>(barrel_ptz_offset_x)/sqrt(xyz[0]*xyz[0] + xyz[2]*xyz[2]));
        if(xyz[0] > 0)
        {
            Beta = atan(-xyz[0]/xyz[2]);
            angle_x = static_cast<float>(-(alpha+Beta)); //camera coordinate
        }else if(xyz[0] < static_cast<double>(barrel_ptz_offset_x))
        {
            Beta = atan(xyz[0]/xyz[2]);
            angle_x = static_cast<float>(-(alpha - Beta));
        }else
        {
            Beta = atan(xyz[0]/xyz[2]);
            angle_x = static_cast<float>(Beta-alpha);   // camera coordinate
        }
    }else{
        angle_x = static_cast<float>(atan2(xyz[0],xyz[2]));
    }
    angle_x = static_cast<float>(angle_x) * 57.2957805f;
    angle_y = static_cast<float>(angle_y) * 57.2957805f;
    dist = static_cast<float>(xyz[2]);
#if SHOW_ANGLE_INFORMATION == 1
    cout << "angle_x:" << angle_x << "     angle_y:" << angle_y << "    dist:" << dist <<endl;
#endif
}

void RM_SolveAngle::draw_Coordinate(Mat & input){
    vector<Point2f> reference_Img;
    vector<Point3f> reference_Obj;
    reference_Obj.clear();
    reference_Obj.push_back(Point3f(0.0, 0.0, 0.0));
    reference_Obj.push_back(Point3f(100, 0.0, 0.0));
    reference_Obj.push_back(Point3f(0.0, 100, 0.0));
    reference_Obj.push_back(Point3f(0.0, 0.0, 100));

    projectPoints(reference_Obj, rvec, tvec, cameraMatrix, distCoeffs, reference_Img);

    line(input, reference_Img[0], reference_Img[1], Scalar(0, 0, 255), 2);
    line(input, reference_Img[0], reference_Img[2], Scalar(0, 255, 0), 2);
    line(input, reference_Img[0], reference_Img[3], Scalar(255, 0, 0), 2);

    //imshow("outImg", input);
}

RM_SolveAngle::~RM_SolveAngle(){
}
