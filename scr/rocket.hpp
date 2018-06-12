//
//  rocket.hpp
//  OpenTsiolkovsky
//
//  Created by Takahiro Inagawa on 2018/05/31.
//  Copyright © 2018 Takahiro Inagawa. All rights reserved.
//

#ifndef rocket_hpp
#define rocket_hpp

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cmath>
#include "../lib/Eigen/Core"
#include "../lib/Eigen/Geometry"
#include "../boost/numeric/odeint.hpp"
#include "../lib/picojson.h"
#include "air.hpp"
#include "Orbit.hpp"
#include "gravity.hpp"
#include "fileio.hpp"
#include "coordinate_transform.hpp"

using namespace Eigen;
using namespace std;


struct RocketStage{
private:
public:
    std::ifstream fin;

    using state = std::array<double, 7>;  // variables of ODE

    string name;
    string flight_mode;
    int num_stage = 0;
    //    ==== flag ====
    bool is_powered = false;
    bool is_separated = false;
    //    ==== calculate ====
    double calc_start_time = 0.0;
    double calc_end_time = 0.0;
    double calc_step_time = 0.01;
    enum EPower_flight_mode {
        _3DoF = 0, _3DoF_with_delay = 1, _6DoF= 2, _6DoF_aerodynami_stable = 3
    };
    enum EFree_flight_mode {
        aerodynamic_stable = 0, _3DoF_defined = 1, ballistic_flight = 2
    };
    EPower_flight_mode power_flight_mode;
    EFree_flight_mode free_flight_mode;
    //    ==== flying body ====
    double mass_init;
    double ballistic_coef = 0.0;
    //    ==== wind ====
    bool wind_file_exist;
    string wind_file_name;
    Vector3d wind_const;
    MatrixXd wind_mat;              // [altitude[m], wind speed[m/s], wind direction[deg]]
    //    ==== initial position & velocity ====
    Vector3d launch_pos_LLH;        // initial position at launch (LLH coordinate) [deg,deg,m]
    Vector3d launch_pos_ECEF;       // initial position at launch (ECEF coordinate) [m]
    Vector3d launch_vel_NED;        // initial velocity at launch (NED coordinate) [m/s]
    Vector3d launch_vel_ECEF;       // initial velocity at launch (ECEF coordinate) [m/s]
    Vector3d posLLH_init;           // initial position when each objects started
    Vector3d vel_NED_init;
    Vector3d posECI_init;
    Vector3d velECI_init;
    //    ==== position & velocity @ separation
    Vector3d posECI_separation;
    Vector3d velECI_separation;
    //    == thrust ==
    bool Isp_file_exist = false;
    string Isp_file_name;
    MatrixXd Isp_mat;
    double Isp_const = 0.0;             // [sec]
    bool thrust_file_exist;
    string thrust_file_name;
    MatrixXd thrust_mat;
    double thrust_const = 0.0;          // [N]
    double burn_start_time;             // [sec]
    double burn_end_time;               // [sec]
    double burn_time;                   // [sec]
    double throat_diameter;             // [m]
    double throat_area;                 // [m2]
    double nozzle_expansion_ratio;      // [-]
    double nozzle_exhaust_pressure;     // [Pa]
    //    == aerodynamics ==
    double body_diameter = 0.0;         // [m]
    double body_area = 0.0;             // body cross-sectional area [m2]
    double CL_const = 0.0;              // Lift coefficient [-]
    bool CL_file_exist;
    string CL_file_name;
    MatrixXd CL_mat;
    double CLa_const = 0.0;             // Lift curve slope [/rad]
    bool CLa_file_exist;
    string CLa_file_name;
    MatrixXd CLa_mat;
    double CD_const = 0.0;              // Drag coefficinet [-]
    bool CD_file_exist;
    string CD_file_name;
    MatrixXd CD_mat;
    //    == attitude ==
    bool attitude_file_exist;
    string attitude_file_name;
    MatrixXd attitude_mat;
    double attitude_azimth_const_deg;   // [deg]
    double attitude_elevation_const_deg;// [deg]
    //    == dumping product ==
    bool dump_exist = false;
    double dump_separation_time = 0.0;
    double dump_mass = 0.0;
    double dump_ballistic_coef = 0.0;
    Vector3d vel_dump_additional_NEDframe;
    //    == stage ==
    bool following_stage_exist = false;
    double previous_stage_separation_time = 0.0;  // [sec]
    double later_stage_separation_time = 1.0e+100;       //[sec]

    //    ==== variables used in ODE ====
    double g0 = 9.80665;
    double thrust = 0.0;
    double thrust_momentum = 0.0;
    double thrust_SL = 0.0;
    double thrust_vac = 0.0;
    double Isp = 0.1;
    double Isp_vac = 0.2;
    double m_dot = 0.0;
    double nozzle_exhaust_area = 0.0;
    double CD = 0.0;
    double CL = 0.0;
    double drag = 0.0;
    double air_density = 0.0;
    double vel_AIR_BODYframe_abs = 0.0;
    double vel_AIR_NEDframe_abs = 0.0;
    double dynamic_pressure = 0.0;
    double force_drag = 0.0;
    double force_lift = 0.0;
    Air air;
    double wind_speed = 0.0;
    double wind_direction = 0.0;
    double azimth = 0;
    double elevation = pi/2;
    double mach_number = 0.0;

    //    ==== loss velocity ====
    double loss_gravity = 0.0;
    double loss_aerodynamics = 0.0;
    double loss_thrust = 0.0;
    double loss_control = 0.0;
    double loss_total = 0.0;

    //    ==== pos/vel vector and direct cosine matrix
    Vector3d posECI_;
    Vector3d velECI_;
    Vector3d accECI_;
    Vector3d accBODY_;
    Matrix3d dcmECI2ECEF_;
    Vector3d posECEF_;
    Vector3d posLLH_;
    Matrix3d dcmECEF2NED_;
    Matrix3d dcmNED2ECEF_;
    Matrix3d dcmECI2NED_;
    Matrix3d dcmNED2ECI_;
    Vector3d vel_ECEF_NEDframe_;
    Vector3d vel_wind_NEDframe_;
    Vector3d vel_AIR_BODYframe_;
    Vector3d vel_AIR_NEDframe_;
    Vector3d vel_BODY_NEDframe_;
    Vector3d attack_of_angle_;
    Matrix3d dcmBODY2AIR_;
    Matrix3d dcmBODY2NED_;
    Matrix3d dcmNED2BODY_;
    Matrix3d dcmECI2BODY_;
    Matrix3d dcmBODY2ECI_;
    Matrix3d dcmECEF2NED_init_;
    Matrix3d dcmECI2NED_init_;

    Vector3d force_air_vector;
    Vector3d force_thrust_vector;
    Vector3d gravity_vector;
    Vector3d posLLH_IIP_;
    double downrange;

    picojson::object source_json_object;

    virtual void operator()(const state& x, state& dx, double t);
    
    RocketStage(){};  // default constractor
    RocketStage& operator=(const RocketStage rocket_stage);
    RocketStage(picojson::object o_each, picojson::object o);
    RocketStage(const RocketStage& rocket_stage, Vector3d posECI_init, Vector3d velECI_init);
    // ↑this is for dumping product constructor

    void update_from_time_and_altitude(double time, double altitude);  // time[s] and altitude[m]
    void update_from_mach_number();
    void progress(double time_now);

    void deep_copy(const RocketStage& obj){
        name = obj.name;
        calc_start_time = obj.calc_start_time;
        calc_end_time = obj.calc_end_time;
        calc_step_time = obj.calc_step_time;
        mass_init = obj.mass_init;
        ballistic_coef = obj.ballistic_coef;
        wind_file_exist = obj.wind_file_exist;
        wind_file_name = obj.wind_file_name;
        wind_const = obj.wind_const;
        wind_mat = obj.wind_mat;
        launch_pos_LLH = obj.launch_pos_LLH;
        launch_pos_ECEF = obj.launch_pos_ECEF;
        launch_vel_NED = obj.launch_vel_NED;
        launch_vel_ECEF = obj.launch_vel_ECEF;
        posLLH_init = obj.posLLH_init;
        vel_NED_init = obj.vel_NED_init;
        posECI_init = obj.posECI_init;
        velECI_init = obj.velECI_init;
        
        is_powered = obj.is_powered;
        is_separated = obj.is_separated;
        power_flight_mode = obj.power_flight_mode;
        free_flight_mode = obj.free_flight_mode;

        num_stage = obj.num_stage;
        
        Isp_file_exist = obj.Isp_file_exist;
        Isp_file_name = obj.Isp_file_name;
        Isp_mat = obj.Isp_mat;
        Isp_const = obj.Isp_const;
        thrust_file_exist = obj.thrust_file_exist;
        thrust_file_name = obj.thrust_file_name;
        thrust_mat = obj.thrust_mat;
        thrust_const = obj.thrust_const;
        burn_start_time = obj.burn_start_time;
        burn_end_time = obj.burn_end_time;
        burn_time = obj.burn_time;
        throat_diameter = obj.throat_diameter;
        throat_area = obj.throat_area;
        nozzle_expansion_ratio = obj.nozzle_expansion_ratio;
        nozzle_exhaust_pressure = obj.nozzle_exhaust_pressure;
        
        body_diameter = obj.body_diameter;
        body_area = obj.body_area;
        CL_const = obj.CL_const;
        CL_file_exist = obj.CL_file_exist;
        CL_file_name = obj.CL_file_name;
        CL_mat = obj.CL_mat;
        CLa_const = obj.CLa_const;
        CLa_file_exist = obj.CLa_file_exist;
        CLa_file_name = obj.CLa_file_name;
        CLa_mat = obj.CLa_mat;
        CD_const = obj.CD_const;
        CD_file_exist = obj.CD_file_exist;
        CD_file_name = obj.CD_file_name;
        CD_mat = obj.CD_mat;
        
        attitude_file_exist = obj.attitude_file_exist;
        attitude_file_name = obj.attitude_file_name;
        attitude_mat = obj.attitude_mat;
        attitude_azimth_const_deg = obj.attitude_azimth_const_deg;
        attitude_elevation_const_deg = obj.attitude_elevation_const_deg;
        
        dump_exist = obj.dump_exist;
        dump_separation_time = obj.dump_separation_time;
        dump_mass = obj.dump_mass;
        dump_ballistic_coef = obj.dump_ballistic_coef;
        vel_dump_additional_NEDframe = obj.vel_dump_additional_NEDframe;
        
        following_stage_exist = obj.following_stage_exist;
        previous_stage_separation_time = obj.previous_stage_separation_time;
        later_stage_separation_time = obj.later_stage_separation_time;
        g0 = obj.g0;
        thrust = obj.thrust;
        thrust_momentum = obj.thrust_momentum;
        thrust_SL = obj.thrust_SL;
        thrust_vac = obj.thrust_vac;
        Isp = obj.Isp;
        Isp_vac = obj.Isp_vac;
        m_dot = obj.m_dot;
        nozzle_exhaust_area = obj.nozzle_exhaust_area;
        CD = obj.CD;
        CL = obj.CL;
        drag = obj.drag;
        air_density = obj.air_density;
        vel_AIR_BODYframe_abs = obj.vel_AIR_BODYframe_abs;
        vel_AIR_NEDframe_abs = obj.vel_AIR_NEDframe_abs;
        dynamic_pressure = obj.dynamic_pressure;
        force_drag = obj.force_drag;
        force_lift = obj.force_lift;
        air = obj.air;
        wind_speed = obj.wind_speed;
        wind_direction = obj.wind_direction;
        azimth = obj.azimth;
        elevation = obj.elevation;
        mach_number = obj.mach_number;
        loss_gravity = obj.loss_gravity;
        loss_aerodynamics = obj.loss_aerodynamics;
        loss_thrust = obj.loss_thrust;
        loss_control = obj.loss_control;
        loss_total = obj.loss_total;
        
        posLLH_ = obj.posLLH_;
    }
    
    RocketStage(const RocketStage& obj){  // copy constractor
        deep_copy(obj);
    }
};

class Rocket{
private:
public:
    std::vector<RocketStage> rs;  // rocket_stages
    std::vector<RocketStage> fo;  // flying_objects
    
    Rocket(string input_filename);
    void flight_simulation();
};


struct CsvObserver : public RocketStage{
    std::ofstream fout;
    string header = "time(s),mass(kg),thrust(N),lat(deg),lon(deg),altitude(m),"
                    "pos_ECI_X(m),pos_ECI_Y(m),pos_ECI_Z(m),"
                    "vel_ECI_X(m/s),vel_ECI_Y(m/s),vel_ECI_Z(m/s),"
                    "vel_NED_X(m/s),vel_NED_Y(m/s),vel_NED_Z(m/s),"
                    "acc_ECI_X(m/s2),acc_ECI_Y(m/s2),acc_ECI_Z(m/s2),"
                    "acc_Body_X(m/s),acc_Body_Y(m/s),acc_Body_Z(m/s),"
                    "Isp(s),Mach number,attitude_azimth(deg),attitude_elevation(deg),"
                    "attack of angle alpha(deg),attack of angle beta(deg),all attack of angle gamma(deg),"
                    "dynamic pressure(Pa),aero Drag(N),aero Lift(N),"
                    "wind speed(m/s),wind direction(deg),downrange(m),"
                    "IIP_lat(deg),IIP_lon(deg),"
                    "dcmBODY2ECI_11,dcmBODY2ECI_12,dcmBODY2ECI_13,"
                    "dcmBODY2ECI_21,dcmBODY2ECI_22,dcmBODY2ECI_23,"
                    "dcmBODY2ECI_31,dcmBODY2ECI_32,dcmBODY2ECI_33,"
                    "loss_gravity(m/s2),"
                    "loss_aerodynamics(m/s2),"
                    "loss_thrust(m/s2),"
                    "is_powered(1=powered 0=free),"
                    "is_separated(1=already 0=still)";

    using RocketStage::RocketStage; // Inheritance constructor
    
    CsvObserver(const std::string& FileName, bool isAddition = false){
        if (isAddition == false){ // 追加書き込みモードかどうか
            fout.open(FileName, std::ios_base::out);
            fout << header << std::endl;
        } else {
            fout.open(FileName, std::ios_base::out | std::ios_base::app); // 追加書き込みモード
        }
    };
    
    virtual void operator()(const state& x, double t);
//    void to_csv(vector<vector<double>> vec);
};


#endif /* rocket_hpp */
