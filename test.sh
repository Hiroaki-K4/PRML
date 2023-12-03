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
fi
