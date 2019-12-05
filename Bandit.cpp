//
// Created by liya on 11/24/19.
//

#include "Bandit.h"
#include <cmath>
#include <random>
#include <algorithm>
#include <exception>
#include <iostream>

Bandit::Bandit(int k_arms, double epsilon, double initial, bool bernoulli, double step_size, bool sample_averages,
               double *ucb, bool gradient, bool gradient_baseline, double true_reward) :
        k_arms(k_arms), epsilon(epsilon), step_size(step_size), sample_averages(sample_averages), ucb(ucb),
        gradient(gradient), gradient_baseline(gradient_baseline), true_reward(true_reward), bernoulli(bernoulli) {
    time_step = 0;
    q_true.resize(k_arms, 0);
    q_est.resize(k_arms, 0);
    average_reward = 0;
    if (!bernoulli) {
        std::normal_distribution<double> distribution(0, 1);
        for (int i = 0; i < k_arms; i++) {
            q_true[i] = (distribution(gen) + true_reward);
            q_est[i] = initial;
            action_count.push_back(0);
        }
    } else {
        for (int i = 0; i < k_arms; i++) {
            std::uniform_real_distribution<double> p_a;
            q_true[i] = (p_a(gen) + true_reward);
            q_est[i] = initial;
            action_count.push_back(0);
        }
    }
    best_action = std::distance(q_true.begin(), std::max_element(q_true.begin(), q_true.end()));
}

int Bandit::get_action() {
    if (epsilon > 0) {
        std::uniform_int_distribution<int> uniform_int_distr(0, k_arms - 1);
        std::binomial_distribution<int> bin_dist(1, epsilon);
        if (bin_dist(gen) == 1) {
            return uniform_int_distr(gen);
        }
    }
    if (ucb != nullptr) {
        auto ucb_est = q_est;
        for (int i = 0; i < k_arms; i++) ucb_est[i] += (*ucb) * sqrt(log(time_step + 1) / (action_count[i] + 1));
        return distance(ucb_est.begin(), max_element(ucb_est.begin(), ucb_est.end()));
    }
    if (gradient) {
        auto exp_est = q_est;
        for (int i = 0; i < k_arms; i++) exp_est[i] = exp(q_est[i]);
        auto tot = accumulate(exp_est.begin(), exp_est.end(), 0.0);
        action_prob = move(exp_est);
        for (auto &x : action_prob) x = x / tot;
        std::discrete_distribution<int> disc_dist(action_prob.begin(), action_prob.end());
        return disc_dist(gen);
    }
    return distance(q_est.begin(), max_element(q_est.begin(), q_est.end()));
}

double Bandit::take_action(int action_index) {
    double reward;
    if (!bernoulli) {
        std::normal_distribution<double> norm_dist(0, 1);
        reward = norm_dist(gen) + q_true[action_index];
    } else {
        std::uniform_real_distribution<double> unif;
        double prob = q_true[action_index];
        double rand = unif(gen);
        reward = rand <= prob ? 1 : 0;
    }
    time_step++;
    average_reward = (time_step - 1.0) / time_step * average_reward + reward / time_step;
    action_count[action_index]++;
    if (sample_averages) {
        q_est[action_index] += 1.0 / action_count[action_index] * (reward - q_est[action_index]);
    } else if (gradient) {
        std::vector<int> is_action(k_arms, 0);
        is_action[action_index] = 1;
        auto baseline = gradient_baseline ? average_reward : 0;
        for (int i = 0; i < k_arms; i++) q_est[i] += step_size * (reward - baseline) * (is_action[i] - action_prob[i]);
    } else {
        q_est[action_index] += step_size * (reward - q_est[action_index]);
    }
    return reward;
}




