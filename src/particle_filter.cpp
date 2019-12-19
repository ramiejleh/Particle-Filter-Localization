/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * Done: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * Done: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  std::default_random_engine gen;
  
  // resize the vectors of particles
  num_particles = 100;
  particles.resize(num_particles);
  
  // create normal distributions for x, y, and theta
  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);
  
  // generate the particles
  for(int i = 0; i < num_particles; i++){
    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);
    particles[i].weight = 1;
  }

  is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * Done: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */

  // Engine for later generation of particles
  std::default_random_engine gen;

  // generate random Gaussian noise
  std::normal_distribution<double> dist_x(0, std_pos[0]);
  std::normal_distribution<double> dist_y(0, std_pos[1]);
  std::normal_distribution<double> dist_theta(0, std_pos[2]);

  for(int i = 0; i < num_particles; i++){
    // add measurements to each particle
    if(fabs(yaw_rate) < 0.0001){  // constant velocity
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);

    } else {
      particles[i].x += velocity / yaw_rate * ( sin( particles[i].theta + yaw_rate*delta_t ) - sin(particles[i].theta) );
      particles[i].y += velocity / yaw_rate * ( cos( particles[i].theta ) - cos( particles[i].theta + yaw_rate*delta_t ) );
      particles[i].theta += yaw_rate * delta_t;
    }

    // predicted particles with added sensor noise
    particles[i].x += dist_x(gen);
    particles[i].y += dist_y(gen);
    particles[i].theta += dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * Done: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  for (unsigned int i = 0; i < observations.size(); i++) { // For each observation

    // Initialize min distance as a high number.
    double min_distance = std::numeric_limits<double>::max();

    for (unsigned j = 0; j < predicted.size(); j++ ) { // For each predition.
      double distance = dist(predicted[j].x, predicted[j].y, observations[i].x, observations[i].y);

      // If the "distance" is less than min, stored the id and update min.
      if ( distance < min_distance ) {
        min_distance = distance;
        observations[i].id = predicted[j].id;
      }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * Done: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  // Iterate through each particle
  for(int i = 0; i < num_particles; i++){
    particles[i].weight = 1.0;

    // Set up landmarks
    vector<LandmarkObs> predictions;
    for(unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++){
      const Map::single_landmark_s landmark = map_landmarks.landmark_list[j];
      double distance = dist(particles[i].x, particles[i].y, landmark.x_f, landmark.y_f);
      if( distance < sensor_range){ // if the landmark is within the sensor range, save it to predictions
        predictions.push_back(LandmarkObs{landmark.id_i, landmark.x_f, landmark.y_f});
      }
    }

    // Convert vehicle observations coordinates to map coordinates
    vector<LandmarkObs> observations_map;
    double cos_theta = cos(particles[i].theta);
    double sin_theta = sin(particles[i].theta);

    for(unsigned int k = 0; k < observations.size(); k++){
      const LandmarkObs observation = observations[k];
      LandmarkObs temp;
      temp.x = observation.x * cos_theta - observation.y * sin_theta + particles[i].x;
      temp.y = observation.x * sin_theta + observation.y * cos_theta + particles[i].y;
      observations_map.push_back(temp);
    }

    // Find landmark index for each observation
    dataAssociation(predictions, observations_map);

    // Update the particle's weight:
    for(unsigned int o = 0; o < observations_map.size(); o++){
      const LandmarkObs observation_map = observations_map[o];
      Map::single_landmark_s landmark = map_landmarks.landmark_list.at(observation_map.id-1);
      double x_term = pow(observation_map.x - landmark.x_f, 2) / (2 * pow(std_landmark[0], 2));
      double y_term = pow(observation_map.y - landmark.y_f, 2) / (2 * pow(std_landmark[1], 2));
      double w = exp(-(x_term + y_term)) / (2 * M_PI * std_landmark[0] * std_landmark[1]);
      particles[i].weight *=  w;
    }

    weights.push_back(particles[i].weight);

  }

}

void ParticleFilter::resample() {
  /**
   * Done: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

  // Vector for new particles
  vector<Particle> new_particles (num_particles);
  
  // Use discrete distribution to return particles by weight
  std::random_device rd;
  std::default_random_engine gen(rd());
  for (int i = 0; i < num_particles; i++) {
    std::discrete_distribution<int> index(weights.begin(), weights.end());
    new_particles[i] = particles[index(gen)];
    
  }
  
  // Replace old particles with the resampled particles
  particles = new_particles;
  weights.clear();
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}