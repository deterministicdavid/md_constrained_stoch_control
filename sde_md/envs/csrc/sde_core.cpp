#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cmath>
#include <algorithm>

namespace py = pybind11;

// Helper: Compute Drift ( b(x, a) = 0.1*x + a )
// We write this inline or as a helper.
void compute_drift(const float* x, const float* a, float* out, int size) {
    for (int i = 0; i < size; i++) {
        out[i] = 0.0f * x[i] + a[i];
    }
}

// Helper: Compute Diffusion ( 0.01 * Identity )
// Since it is diagonal constant, we can optimize the multiply directly.
// But to be generic, let's assume we return the matrix values or handle the multiply.
// OPTIMIZATION: directly compute diffusion @ dW.
// If sigma = 0.01 * I, then sigma * dW = 0.01 * dW
void compute_diffusion_term(const float* dW, const float sigma, float* out, int size) {
    for (int i = 0; i < size; i++) {
        out[i] = sigma * dW[i];
    }
}

// Cost Functions
float compute_running_cost(const float* x, const float* a, int size, const float control_cost_factor) {
    //float sum_x2 = 0.0f;
    float sum_a2 = 0.0f;
    for (int i = 0; i < size; i++) {
        //sum_x2 += x[i] * x[i];
        sum_a2 += a[i] * a[i];
    }
    //return sum_x2 + 0.01f * sum_a2;
    return control_cost_factor * sum_a2;
}

float compute_terminal_cost(const float* x, int size, const float space_cost_factor) {
    float sum_x2 = 0.0f;
    for (int i = 0; i < size; i++) {
        sum_x2 += x[i] * x[i];
    }
    return space_cost_factor * sum_x2;
}

// The Main Step Function
std::tuple<py::array_t<float>, float, float, bool, bool> step_sde_cpp(
    py::array_t<float> state_array,
    py::array_t<float> action_array,
    py::array_t<float> dW_array,
    float dt,
    float sigma,
    float radius,
    float current_time,
    float T,
    float control_cost_factor,
    float space_cost_factor_terminal
) {
    // 1. Request Buffer Info (pointers to raw data)
    // This is ZERO-COPY access to the Numpy arrays.
    py::buffer_info state_buf = state_array.request();
    py::buffer_info action_buf = action_array.request();
    py::buffer_info dW_buf = dW_array.request();

    float* x = static_cast<float*>(state_buf.ptr);
    float* a = static_cast<float*>(action_buf.ptr);
    float* dw = static_cast<float*>(dW_buf.ptr);

    int size = state_buf.shape[0]; // Dimension

    // 2. Allocations for intermediate steps (on stack if small, or heap)
    // For safety with variable size, we use std::vector
    std::vector<float> drift(size);
    std::vector<float> diffusion_term(size);
    
    // Compute Drift & Diffusion
    compute_drift(x, a, drift.data(), size);
    compute_diffusion_term(dw, sigma, diffusion_term.data(), size);

    // 3. Euler-Maruyama Update & Output Allocation
    // X_new = X + drift*dt + diff*sqrt(dt)
    auto next_state_array = py::array_t<float>(size);
    py::buffer_info next_buf = next_state_array.request();
    float* next_x = static_cast<float*>(next_buf.ptr);

    float sqrt_dt = std::sqrt(dt);
    float dist_sq = 0.0f;

    for (int i = 0; i < size; i++) {
        next_x[i] = x[i] + drift[i] * dt + diffusion_term[i] * sqrt_dt;
        dist_sq += next_x[i] * next_x[i];
    }

    // 4. Update Time
    float next_time = current_time + dt;

    // 5. Check Constraints
    bool is_outside = dist_sq > (radius * radius);
    bool time_limit = next_time >= T;
    
    bool terminated = false;
    bool truncated = false;
    float reward = 0.0f;

    if (is_outside || time_limit) {
        terminated = true; // Simplified logic: exit or timeout both end episode
        if (time_limit && !is_outside) truncated = true; // Distinguish timeout
        
        // Terminal Cost
        reward = -compute_terminal_cost(x, size, space_cost_factor_terminal); 
    } else {
        // Running Cost
        float cost = compute_running_cost(x, a, size, control_cost_factor);
        reward = -cost * dt;
    }

    return std::make_tuple(next_state_array, reward, next_time, terminated, truncated);
}

// Module Definition
PYBIND11_MODULE(sde_cpp, m) {
    m.doc() = "Optimized C++ SDE implementation for Gymnasium";
    m.def("step_sde", &step_sde_cpp, "Run one Euler-Maruyama step");
}