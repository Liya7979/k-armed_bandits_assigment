//
// Created by liya on 11/24/19.
//

#ifndef K_ARMED_BANDITS_BANDIT_H
#define K_ARMED_BANDITS_BANDIT_H


#include <vector>
#include <random>

class Bandit {
private:
    int k_arms;
    double step_size;
    bool gradient;
    bool gradient_baseline;
    double time_step;
    double *ucb;
    double average_reward;
    double true_reward;
    const double *p_a_ptr;
    std::vector<double> action_prob_;
    std::default_random_engine rnd_engine;
    std::vector<double> q_true;   // real reward for each action
    std::vector<double> q_est;    // estimated reward for each action
    std::vector<int> action_count; // count of chosen for each action
public:

    Bandit(int k_arms=10, double epsilon = 0., double initial = 0., const double *p_a = nullptr, double step_size = 0.1, bool sample_averages = false,
           double *ucb = nullptr, bool gradient = false, bool gradient_baseline = false, double true_reward = 0.);

//    Bandit(const Bandit& other);

    int action();

    double step(int action_index);

    double epsilon{};
    bool sample_averages{};
    int best_action;
};


#endif //K_ARMED_BANDITS_BANDIT_H
