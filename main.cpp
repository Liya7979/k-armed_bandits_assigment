#include <iostream>
#include <vector>
#include <fstream>
#include "Bandit.h"


void clear_bandits(const std::vector<std::vector<Bandit *>> &bandits);

void send_to_file(const std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> &res,
                  const std::string &params);

std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>
bandit_simulation(int num_bandits, int time_step, std::vector<std::vector<Bandit *>> &bandits) {
    std::vector<std::vector<double>> best_action_counts(bandits.size(), std::vector<double>(time_step, 0.0));
    std::vector<std::vector<double>> average_rewards(bandits.size(), std::vector<double>(time_step, 0.0));
    for (size_t b = 0; b < bandits.size(); b++) {
        for (int i = 0; i < num_bandits; i++) {
            for (int t = 0; t < time_step; t++) {
                auto action_index = bandits[b][i]->action();
                auto reward = bandits[b][i]->step(action_index);
                average_rewards[b][t] += reward;
                if (action_index == bandits[b][i]->best_action) {
                    best_action_counts[b][t] += 1;
                }
            }
        }
        for (auto &x : best_action_counts[b]) {
            x = x / num_bandits;
        }
        for (auto &x : average_rewards[b]) {
            x = x / num_bandits;
        }

    }
    return make_pair(best_action_counts, average_rewards);
}

void epsilon_greedy(int num_bandits, int time_step, int k_arms, bool bernoulli) {
    std::vector<double> epsilons = {0, 0.1, 0.01};
    std::vector<std::vector<Bandit *>> bandits;
    for (double epsilon : epsilons) {
        std::vector<Bandit *> vec_bandits;
        for (int j = 0; j < num_bandits; j++) {
            vec_bandits.push_back(new Bandit(k_arms, epsilon, 0., bernoulli));
            vec_bandits.back()->epsilon = epsilon;
            vec_bandits.back()->sample_averages = true;
        }
        bandits.push_back(vec_bandits);
    }
    auto res = bandit_simulation(num_bandits, time_step, bandits);
    std::string filename = "epsilon_" + std::to_string(k_arms);
    send_to_file(res, filename);
    clear_bandits(bandits);
}


void clear_bandits(const std::vector<std::vector<Bandit *>> &bandits) {
    for (auto &v : bandits)
        for (auto p : v)
            delete p;
}

void optimistic_initial_values(int num_bandits, int time_step, int k_arms) {
    using namespace std;
    vector<vector<Bandit *>> bandits;
    vector<Bandit *> vec_bandits;
    for (int j = 0; j < num_bandits; j++) {
        vec_bandits.push_back(new Bandit(k_arms, 0., 5, false, 0.1));
    }


    bandits.push_back(vec_bandits);

    vec_bandits.clear();

    for (int j = 0; j < num_bandits; j++) {
        vec_bandits.push_back(new Bandit(k_arms, 0.1, 0, false, 0.1));
    }

    bandits.push_back(vec_bandits);

    auto res = bandit_simulation(num_bandits, time_step, bandits);

    std::string filename = "optimistic_" + std::to_string(k_arms);
    send_to_file(res, filename);
    clear_bandits(bandits);
}

void ucb(int num_bandits, int time_step, int k_arms) {
    std::vector<std::vector<Bandit *>> bandits;
    std::vector<Bandit *> vec_bandits;
    double ucb_param = 2;
    for (int j = 0; j < num_bandits; j++) {
        vec_bandits.push_back(new Bandit(k_arms, 0., 0., false,
                                         0.1, false, &ucb_param));
    }
    bandits.push_back(vec_bandits);
    vec_bandits.clear();
    for (int j = 0; j < num_bandits; j++) {
        vec_bandits.push_back(new Bandit(k_arms, 0.1, 0., false,
                                         0.1));
    }
    bandits.push_back(vec_bandits);
    auto res = bandit_simulation(num_bandits, time_step, bandits);
    std::string filename = "ucb_" + std::to_string(k_arms);
    send_to_file(res, filename);
    clear_bandits(bandits);
}

void gradient_bandit(int num_bandits, int time_step, int k_arms) {
    std::vector<std::vector<Bandit *>> bandits;
    std::vector<Bandit *> vec_bandits;
    for (int j = 0; j < num_bandits; j++) {
        vec_bandits.push_back(new Bandit(k_arms, 0., 0., false,
                                         0.1, false, nullptr,
                                         true, true, 4));
    }
    bandits.push_back(vec_bandits);
    vec_bandits.clear();
    for (int j = 0; j < num_bandits; j++) {
        vec_bandits.push_back(new Bandit(k_arms, 0., 0., false,
                                         0.1, false, nullptr,
                                         true, false, 4));
    }
    bandits.push_back(vec_bandits);
    vec_bandits.clear();
    for (int j = 0; j < num_bandits; j++) {
        vec_bandits.push_back(new Bandit(k_arms, 0., 0., false,
                                         0.4, false, nullptr,
                                         true, true, 4));
    }
    bandits.push_back(vec_bandits);
    vec_bandits.clear();
    for (int j = 0; j < num_bandits; j++) {
        vec_bandits.push_back(new Bandit(k_arms, 0., 0., false,
                                         0.4, false, nullptr,
                                         true, false, 4));
    }
    bandits.push_back(vec_bandits);
    auto res = bandit_simulation(num_bandits, time_step, bandits);
    std::string filename = "gradient_" + std::to_string(k_arms);
    send_to_file(res, filename);
    clear_bandits(bandits);
}

void send_to_file(const std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> &res,
                  const std::string &params) {
    std::ofstream output;
    std::string name = "bandit_" + params;
    output.open(name);
    for (auto &v : res.first) {
        for (auto &u: v) {
            output << u << " ";
        }
        output << std::endl;
    }
    output << "\n\n";
    for (auto &v : res.second) {
        for (auto &u: v) {
            output << u << " ";
        }
        output << std::endl;
    }
    output << "\n" << std::endl;
    output.close();
}

int main() {
    int arms[4] = {5, 10, 20, 40};
    for (auto &arm: arms) {
        epsilon_greedy(10000, 1000, arm, true);
        //std::cout << p_a << std::endl;
//        optimistic_initial_values(10000, 1000, arm);
//        ucb(10000, 1000, arm);
//        gradient_bandit(10000, 1000, arm);
        break;

    }
    return 0;
}