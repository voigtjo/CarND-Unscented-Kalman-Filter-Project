#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::ArrayXd;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;
using std::cout;
using std::endl;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {

	// if this is false, laser measurements will be ignored (except during init)
	use_laser_ = true;

	// if this is false, radar measurements will be ignored (except during init)
	use_radar_ = true;

	// initial state vector
	x_ = VectorXd::Zero(n_x_);

	// initial covariance matrix
	P_ = MatrixXd::Zero(n_x_,   n_x_);

	Q_ = MatrixXd::Zero(2, 2); // Process Noise Covariance Matrix


	// Process noise standard deviation longitudinal acceleration in m/s^2
	std_a_ = 0.4; // default 30

	// Process noise standard deviation yaw acceleration in rad/s^2
	std_yawdd_ = 0.5; // default 30

	// Laser measurement noise standard deviation position1 in m
	std_laspx_ = 0.15;

	// Laser measurement noise standard deviation position2 in m
	std_laspy_ = 0.15;

	// Radar measurement noise standard deviation radius in m
	std_radr_ = 0.3;

	// Radar measurement noise standard deviation angle in rad
	std_radphi_ = 0.03;

	// Radar measurement noise standard deviation radius change in m/s
	std_radrd_ = 0.3;

	/**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
	 */

	is_initialized_ = false;

	//set state dimension
	int n_x = 5;

	//set augmented dimension
	int n_aug = 7;

	// augmentation dimension
	n_sig_ = 2 * n_aug + 1;

	//set radar measurement dimension
	n_z_radar_ = 3;
	//set lidar measurement dimension
	n_z_lidar_ = 2;

	//define spreading parameter
	lambda_ = 3 - n_aug;

	x_ = VectorXd::Zero(n_x_);
	P_ = MatrixXd::Zero(n_x_, n_x_);
	Xsig_pred_ = MatrixXd::Zero(n_x_, n_sig_);
	weights_ = VectorXd::Zero(n_sig_);
	R_lidar_ = MatrixXd::Zero(n_z_lidar_, n_z_lidar_);
	R_radar_ = MatrixXd::Zero(n_z_radar_, n_z_radar_);

}


UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
	/**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
	 */
	// Initialisation
	if(!is_initialized_)
	{
		float px = 0, py = 0, v = 0;
		if (MeasurementPackage::RADAR == meas_package.sensor_type_) {

			// extract the RADAR measurements and convert from
			// Polar to Cartesian coordinates
			float r = meas_package.raw_measurements_[0];
			float phi = meas_package.raw_measurements_[1];
			float r_dot = meas_package.raw_measurements_[2];

			// calculate position and velocity
			px = r * cos(phi);
			py = r * sin(phi);
			v = r_dot;

		} else if (MeasurementPackage::LASER == meas_package.sensor_type_) {

			// if it is laser, just grab the raw x, y coordinates
			px = meas_package.raw_measurements_[0];
			py = meas_package.raw_measurements_[1];
		}

		x_ << px , py , v, 0, 0;
		P_ << 	1, 0, 0, 0, 0,
				0, 1, 0, 0, 0,
				0, 0, 100, 0, 0,
				0, 0, 0, 100, 0,
				0, 0, 0, 0, 1;

		// set weights
		weights_.fill(0.5 / (n_aug_ + lambda_));
		weights_(0) = lambda_ / (lambda_ + n_aug_);

		//add radar measurement noise covariance matrix
		R_radar_ << std_radr_ * std_radr_ , 0, 0,
				0, std_radphi_ * std_radphi_, 0,
				0, 0, std_radrd_ * std_radrd_;

		//add lidar measurement noise covariance matrix
		R_lidar_ << std_laspx_ * std_laspx_ , 0,
				0, std_laspy_ * std_laspy_;

		time_us_ = meas_package.timestamp_;
		is_initialized_ = true;
	}

	// prediction
	// elapsed seconds
	float dt = (meas_package.timestamp_ - time_us_) / 1000000.0;
	time_us_ = meas_package.timestamp_;

	Prediction(dt);


	// update
	if (use_radar_ && MeasurementPackage::RADAR == meas_package.sensor_type_) {
		UpdateRadar(meas_package);
	}
	else if (use_laser_ && MeasurementPackage::LASER == meas_package.sensor_type_) {
		UpdateLidar(meas_package);
	}

}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
	/**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
	 */

	// augmentation vector
	VectorXd x_aug = VectorXd::Zero(n_aug_);

	// augmented state covariance
	MatrixXd P_aug = MatrixXd::Zero(n_aug_, n_aug_);

	// process noise covariance
	MatrixXd Q = MatrixXd::Zero(n_aug_ - n_x_, n_aug_ - n_x_);
	Q << std_a_ * std_a_, 0, 0, std_yawdd_ * std_yawdd_;

	// sigma points matrix
	MatrixXd Xsig_aug = MatrixXd::Zero(n_aug_, n_sig_);

	//create augmented mean state
	x_aug.head(5) = x_;
	x_aug(5) = 0;
	x_aug(6) = 0;

	//create augmented covariance matrix
	P_aug.fill(0.0);
	P_aug.topLeftCorner(n_x_, n_x_) = P_;
	P_aug.bottomRightCorner(n_aug_ - n_x_, n_aug_ - n_x_) = Q;

	//create square root matrix
	MatrixXd L = P_aug.llt().matrixL();

	//create augmented sigma points
	Xsig_aug.col(0)  = x_aug;
	for (int i = 0; i< n_aug_; i++)
	{
		Xsig_aug.col(i+1)        = x_aug + sqrt(lambda_ + n_aug_) * L.col(i);
		Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * L.col(i);
	}

	//predict sigma points
	for (int i = 0; i< 2*n_aug_+1; i++)
	{
		//extract values for better readability
		double p_x = Xsig_aug(0,i);
		double p_y = Xsig_aug(1,i);
		double v = Xsig_aug(2,i);
		double yaw = Xsig_aug(3,i);
		double yawd = Xsig_aug(4,i);
		double nu_a = Xsig_aug(5,i);
		double nu_yawdd = Xsig_aug(6,i);

		//predicted state values
		double px_p, py_p;

		//avoid division by zero
		if (fabs(yawd) > 0.001) {
			px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
			py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
		}
		else {
			px_p = p_x + v*delta_t*cos(yaw);
			py_p = p_y + v*delta_t*sin(yaw);
		}

		double v_p = v;
		double yaw_p = yaw + yawd*delta_t;
		double yawd_p = yawd;

		//add noise
		px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
		py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
		v_p = v_p + nu_a*delta_t;

		yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
		yawd_p = yawd_p + nu_yawdd*delta_t;

		//write predicted sigma point into right column
		Xsig_pred_(0,i) = px_p;
		Xsig_pred_(1,i) = py_p;
		Xsig_pred_(2,i) = v_p;
		Xsig_pred_(3,i) = yaw_p;
		Xsig_pred_(4,i) = yawd_p;
	}

	// predict mean and convariance matrix
	// set weights
	double weight_0 = lambda_/(lambda_+n_aug_);
	weights_(0) = weight_0;
	for (int i=1; i<2*n_aug_+1; i++) {  //2n+1 weights
		double weight = 0.5/(n_aug_+lambda_);
		weights_(i) = weight;
	}

	//predicted state mean
	x_.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
		x_ = x_+ weights_(i) * Xsig_pred_.col(i);
	}

	//predicted state covariance matrix
	P_.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points

		// state difference
		VectorXd x_diff = Xsig_pred_.col(i) - x_;
		//angle normalization
		while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
		while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

		P_ = P_ + weights_(i) * x_diff * x_diff.transpose() ;
	}
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
	/**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
	 */
	MatrixXd Zsig = MatrixXd::Zero(n_z_lidar_, n_sig_);
	VectorXd z_pred = VectorXd::Zero(n_z_lidar_);
	MatrixXd S = MatrixXd::Zero(n_z_lidar_, n_z_lidar_);

	for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

		// extract values for better readibility
		double p_x = Xsig_pred_(0,i);
		double p_y = Xsig_pred_(1,i);

		// measurement model
		Zsig(0,i) = p_x;
		Zsig(1,i) = p_y;
	}

	//mean predicted measurement
	z_pred.fill(0.0);
	for (int i=0; i < 2*n_aug_+1; i++) {
		z_pred = z_pred + weights_(i) * Zsig.col(i);
	}

	//measurement covariance matrix S
	S.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
		//residual
		VectorXd z_diff = Zsig.col(i) - z_pred;

		//angle normalization
		while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
		while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

		S = S + weights_(i) * z_diff * z_diff.transpose();
	}

	//add measurement noise covariance matrix
	S = S + R_lidar_;

	MatrixXd Tc = MatrixXd::Zero(n_x_, n_z_lidar_);
	Tc.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

		//residual
		VectorXd z_diff = Zsig.col(i) - z_pred;
		//angle normalization
		while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
		while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

		// state difference
		VectorXd x_diff = Xsig_pred_.col(i) - x_;
		//angle normalization
		while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
		while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

		Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
	}

	//Kalman gain K;
	MatrixXd K = Tc * S.inverse();

	//residual
	VectorXd z = VectorXd(n_z_lidar_);
	z << 	meas_package.raw_measurements_[0],
			meas_package.raw_measurements_[1]
	VectorXd z_diff = z - z_pred;

	//angle normalization
	while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
	while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

	//update state mean and covariance matrix
	x_ = x_ + K * z_diff;
	P_ = P_ - K*S*K.transpose();

	NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
	/**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
	 */
	//transform sigma points into measurement space
	MatrixXd Zsig = MatrixXd::Zero(n_z_radar_, n_sig_);
	VectorXd z_pred = VectorXd(n_z_radar_);
	MatrixXd S = MatrixXd(n_z_radar_,n_z_radar_);

	for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

		// extract values for better readibility
		double p_x = Xsig_pred_(0,i);
		double p_y = Xsig_pred_(1,i);
		double v  = Xsig_pred_(2,i);
		double yaw = Xsig_pred_(3,i);

		double v1 = cos(yaw)*v;
		double v2 = sin(yaw)*v;

		// measurement model
		Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y);                        //r
		Zsig(1,i) = atan2(p_y,p_x);                                 //phi
		Zsig(2,i) = (p_x*v1 + p_y*v2 ) / sqrt(p_x*p_x + p_y*p_y);   //r_dot
	}

	//mean predicted measurement
	z_pred.fill(0.0);
	for (int i=0; i < 2*n_aug_+1; i++) {
		z_pred = z_pred + weights_(i) * Zsig.col(i);
	}

	//measurement covariance matrix S
	S.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
		//residual
		VectorXd z_diff = Zsig.col(i) - z_pred;

		//angle normalization
		while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
		while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

		S = S + weights_(i) * z_diff * z_diff.transpose();
	}

	//add measurement noise covariance matrix
	S = S + R_radar_;

	MatrixXd Tc = MatrixXd::Zero(n_x_, n_z_radar_);
	Tc.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

		//residual
		VectorXd z_diff = Zsig.col(i) - z_pred;
		//angle normalization
		while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
		while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

		// state difference
		VectorXd x_diff = Xsig_pred_.col(i) - x_;
		//angle normalization
		while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
		while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

		Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
	}

	//Kalman gain K;
	MatrixXd K = Tc * S.inverse();

	//residual
	VectorXd z = VectorXd(n_z_radar_);
	z << 	meas_package.raw_measurements_[0],
			meas_package.raw_measurements_[1],
			meas_package.raw_measurements_[2];
	VectorXd z_diff = z - z_pred;

	//angle normalization
	while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
	while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

	//update state mean and covariance matrix
	x_ = x_ + K * z_diff;
	P_ = P_ - K*S*K.transpose();

	NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;

}

