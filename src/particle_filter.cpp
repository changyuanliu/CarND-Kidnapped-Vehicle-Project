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
using std::max_element;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 100;  // TODO: Set the number of particles
  std::default_random_engine generator;
  std::normal_distribution<double> xd(x,std[0]);
  std::normal_distribution<double> yd(y,std[1]);
  std::normal_distribution<double> td(theta,std[2]);

  particles.resize(num_particles);
  for(int i=0; i<num_particles; i++)
  {
    particles[i].id = i;
    particles[i].x = x + xd(generator);
    particles[i].y = y + yd(generator);
    particles[i].theta = theta + td(generator);
    particles[i].weight = 1.0;
  }
  is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  std::default_random_engine generator;
  std::normal_distribution<double> xd(0,std_pos[0]);
  std::normal_distribution<double> yd(0,std_pos[1]);
  std::normal_distribution<double> td(0,std_pos[2]);
  for(int i=0; i<num_particles; i++)
  {
    double p_theta = particles[i].theta;
    //deal with yaw_rate==0
    if(fabs(yaw_rate)<1.0e-6)
    {
      particles[i].x += velocity*delta_t*cos(p_theta) + xd(generator);
      particles[i].y += velocity*delta_t*sin(p_theta) + yd(generator);
      particles[i].theta += td(generator);
    }
    else
    {
      //predict per "Calculate Prediction Step: Quiz"
      particles[i].x += velocity/yaw_rate*(sin(p_theta+yaw_rate*delta_t)-sin(p_theta)) + xd(generator);
      particles[i].y += velocity/yaw_rate*(cos(p_theta)-cos(p_theta+yaw_rate*delta_t)) + yd(generator);
      particles[i].theta += yaw_rate*delta_t + td(generator);
    }
  }  

}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  for(int i=0; i<observations.size(); i++)
  {
    //find the closest prediction to each observation
    double min_dist = 1.0e6;
    LandmarkObs min_p;
    for(int j=0; j<predicted.size(); j++)
    {
      double curr_dist = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
      if( curr_dist < min_dist)
      {
        min_dist = curr_dist;
        min_p = predicted[j];
      }
    }
    //Assign the closest precdiction to the observation
    observations[i].id = min_p.id;  
  }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
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
  
  for(int i=0; i<num_particles; i++)
  {    
    //convert the observations to map's coordinate system
    //according to "Quiz: Landmarks" in Lesson 5
    double p_x = particles[i].x;
    double p_y = particles[i].y;
    double p_theta = particles[i].theta;
    vector<LandmarkObs> map_observations = observations;
    for(int j=0; j<observations.size(); j++)
    { 
      double o_x = observations[j].x;
      double o_y = observations[j].y;
      map_observations[j].x = cos(p_theta)*o_x - sin(p_theta)*o_y + p_x;
      map_observations[j].y = sin(p_theta)*o_x + cos(p_theta)*o_y + p_y;
    }
    //find predictions (map_landmarks) in the particle's sensing range
    vector<LandmarkObs> predictions;
    LandmarkObs prdt;
    for(int j=0; j<map_landmarks.landmark_list.size(); j++)
    {
      double lm_dist = dist(p_x, p_y, map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f);
      if( lm_dist <= sensor_range)
      {
        prdt.id = map_landmarks.landmark_list[j].id_i;
        prdt.x = map_landmarks.landmark_list[j].x_f;
        prdt.y = map_landmarks.landmark_list[j].y_f;
        predictions.push_back(prdt);
      }
    }
    //associate the observations with predictions
    dataAssociation(predictions, map_observations);

    //update weights based on multivariate Guassian distribution
    //init weight
    double weight = 1.0;
    for(int j=0; j<map_observations.size(); j++)
    {
      //locate the associated prediction
      LandmarkObs associated_prediction;
      for(int k=0; k<predictions.size(); k++)
      {
        if(predictions[k].id == map_observations[j].id)
        {
          associated_prediction = predictions[k];  
        }
      }
      //calculate 2-d normal distribution for each observation
      double delta_x = map_observations[j].x - associated_prediction.x;
      double delta_y = map_observations[j].y - associated_prediction.y;
      double std_x = std_landmark[0];
      double std_y = std_landmark[1];
      double prob = exp(-0.5*(delta_x*delta_x/(std_x*std_x)+delta_y*delta_y/(std_y*std_y))) / (2*M_PI*std_x*std_y);
      weight *= prob;
    }    
    //update weight in the particle
    particles[i].weight = weight;
    //update weights which will be used in resample later
    //weights.push_back(weight);
  }  
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  
  std::default_random_engine generator;
  std::uniform_int_distribution<int> uni_int_dist(0, num_particles-1);  
  auto index = uni_int_dist(generator);

  //get all of the weights
  vector<double> weights;
  for (int i = 0; i < num_particles; i++) {
    weights.push_back(particles[i].weight);
  }
  //find the max weight
  double max_weight = *max_element(weights.begin(), weights.end());
  std::cout<<"==========resample==========="<<std::endl;
  std::cout<<"max weignt = "<<weights[1]<<std::endl;
  std::uniform_real_distribution<double> uni_real_dist(0.0, 2*max_weight);
  double beta = 0.0;
  
  vector<Particle> resampled_particles;
  //implement the resample wheel
  for(int i=0; i<num_particles; i++)
  {
    beta += uni_real_dist(generator);
    while(particles[index].weight < beta)
    {
      beta -= particles[index].weight;
      index = (index+1) % num_particles;
    }
    resampled_particles.push_back(particles[index]);
  }
  particles = resampled_particles;
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