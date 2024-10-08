#!/bin/bash

COLOR_RESET="\033[m"
COLOR_RED="\033[31m"
COLOR_GREEN="\033[32m"

check_result() {
	if [ $? -ne 0 ]; then
		printf "${COLOR_RED}%s:%s %s${COLOR_RESET}\n" "$1" "$2" ' [ERROR]'
		exit 1
	fi
	printf "${COLOR_GREEN}%s:%s %s${COLOR_RESET}\n" "$1" "$2" ' [OK]'
}

test_probability_distribution() {
    cd probability_distribution
    python3 draw_3d_trigonometric_graph.py NotShow
    check_result "draw_3d_trigonometric_graph.py"
    python3 draw_contour_map.py NotShow
    check_result "draw_contour_map.py"
    python3 draw_trigonometric_graph.py NotShow
    check_result "draw_trigonometric_graph.py"
    python3 draw_multivariate_gaussian_distribution.py NotShow
    check_result "draw_multivariate_gaussian_distribution.py"
    python3 draw_conditional_gaussian_distribution.py NotShow
    check_result "draw_conditional_gaussian_distribution.py"
    python3 draw_students_t_distribution.py NotShow
    check_result "draw_students_t_distribution.py"
    python3 draw_von_mises_distribution.py NotShow
    check_result "draw_von_mises_distribution.py"
    python3 draw_mixture_of_gaussians.py NotShow
    check_result "draw_mixture_of_gaussians.py"
    cd ../
}

test_draw_beta_distribution() {
    cd probability_distribution
    python3 draw_beta_distribution.py
    check_result "draw_beta_distribution.py"
    cd ../
}

test_draw_binomial_distribution() {
    cd probability_distribution
    python3 draw_binomial_distribution.py
    check_result "draw_binomial_distribution.py"
    cd ../
}

test_draw_dirichlet_distribution() {
    cd probability_distribution
    python3 draw_dirichlet_distribution.py
    check_result "draw_dirichlet_distribution.py"
    cd ../
}

test_draw_gaussian_distribution() {
    cd probability_distribution
    python3 draw_gaussian_distribution.py
    check_result "draw_gaussian_distribution.py"
    python3 draw_gaussian_distribution_with_known_variance.py
    check_result "draw_gaussian_distribution_with_known_variance.py"
    python3 draw_gaussian_distribution_with_known_mean.py
    check_result "draw_gaussian_distribution_with_known_mean.py"
    cd ../
}

test_linear_regression_model() {
    cd linear_regression_model
    python3 draw_polynomial_curve_fitting.py NotShow
    check_result "draw_polynomial_curve_fitting.py"
    python3 draw_basis_functions.py NotShow
    check_result "draw_basis_functions.py"
    python3 draw_bias_variance_decomposition.py NotShow
    check_result "draw_bias_variance_decomposition.py"
    python3 draw_sequential_bayesian_learning.py NotShow
    check_result "draw_sequential_bayesian_learning.py"
    python3 draw_predicted_distribution.py NotShow
    check_result "draw_predicted_distribution.py"
    python3 draw_equivalent_kernel.py NotShow
    check_result "draw_equivalent_kernel.py"
    python3 evaluate_evidence_function.py NotShow
    check_result "evaluate_evidence_function.py"
    python3 maximize_evidence_function.py
    check_result "maximize_evidence_function.py"
    cd ../
}

test_linear_discriminative_model() {
    cd linear_discriminative_model
    python3 fishers_linear_discriminant.py NotShow
    check_result "fishers_linear_discriminant.py"
    python3 stochastic_generative_model.py NotShow
    check_result "stochastic_generative_model.py"
    cd ../
}

test_mixture_density_network() {
    cd neural_network
    python3 draw_hyperbolic_functions.py NotShow
    check_result "draw_hyperbolic_functions.py"
    python3 compare_tanh_sigmoid_and_sigmoid.py NotShow
    check_result "compare_tanh_sigmoid_and_sigmoid.py"
    python3 solve_forward_problem.py NotShow
    check_result "solve_forward_problem.py"
    python3 solve_inverse_problem_with_mixture_density_network.py NotShow
    check_result "solve_inverse_problem_with_mixture_density_network.py"
    python3 solve_inverse_problem_with_simple_network.py NotShow
    check_result "solve_inverse_problem_with_simple_network.py"
    cd ../
}

test_kernel_method() {
    cd kernel_method
    python3 draw_kernel_function.py NotShow
    check_result "draw_kernel_function.py"
    python3 nadaraya_watson_model.py NotShow
    check_result "nadaraya_watson_model.py"
    python3 draw_sample_from_gaussian_process_prior.py NotShow
    check_result "draw_sample_from_gaussian_process_prior.py"
    python3 draw_gaussian_process.py NotShow
    check_result "draw_gaussian_process.py"
    cd ../
}

test_kernel_machine() {
    cd kernel_machine/svm
    python3 svm.py NotShow
    check_result "svm.py"
    cd ../../
}

test_graphical_model() {
    cd graphical_model
    python3 remove_noise_using_graphical_model.py
    check_result "remove_noise_using_graphical_model.py"
    cd ../../
}

test_mixed_models_and_EM() {
    cd mixed_models_and_EM
    python3 image_segmentation_by_k-means.py
    check_result "image_segmentation_by_k-means.py"
    cd ../../
}

python3 -m pip install -r requirements.txt

if [ $# -eq 1 ]; then
    if [ $1 = "probability_distribution" ]; then
        test_probability_distribution
    elif [ $1 = "draw_beta_distribution" ]; then
        test_draw_beta_distribution
    elif [ $1 = "draw_binomial_distribution" ]; then
        test_draw_binomial_distribution
    elif [ $1 = "draw_dirichlet_distribution" ]; then
        test_draw_dirichlet_distribution
    elif [ $1 = "draw_gaussian_distribution" ]; then
        test_draw_gaussian_distribution
    elif [ $1 = "linear_regression_model" ]; then
        test_linear_regression_model
    elif [ $1 = "linear_discriminative_model" ]; then
        test_linear_discriminative_model
    elif [ $1 = "mixture_density_network" ]; then
        test_mixture_density_network
    elif [ $1 = "kernel_method" ]; then
        test_kernel_method
    elif [ $1 = "kernel_machine" ]; then
        test_kernel_machine
    elif [ $1 = "graphical_model" ]; then
        test_graphical_model
    elif [ $1 = "mixed_models_and_EM" ]; then
        test_mixed_models_and_EM
    else
        echo "Argument is wrong"
        exit 1
    fi

else
    test_probability_distribution
    test_draw_beta_distribution
    test_draw_binomial_distribution
    test_draw_dirichlet_distribution
    test_draw_gaussian_distribution
    test_linear_regression_model
    test_linear_discriminative_model
    test_mixture_density_network
    test_kernel_method
    test_kernel_machine
    test_graphical_model
    test_mixed_models_and_EM
fi
