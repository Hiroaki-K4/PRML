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
    python3 draw_beta_distribution.py
    check_result "draw_beta_distribution.py"
    python3 draw_binomial_distribution.py
    check_result "draw_binomial_distribution.py"
    python3 draw_contour_map.py NotShow
    check_result "draw_contour_map.py"
    python3 draw_dirichlet_distribution.py
    check_result "draw_dirichlet_distribution.py"
    python3 draw_trigonometric_graph.py NotShow
    check_result "draw_trigonometric_graph.py"
    python3 draw_gaussian_distribution.py
    check_result "draw_gaussian_distribution.py"
    cd ../
}

python3 -m pip install -r requirements.txt

if [ $# -eq 1 ]; then
    if [ $1 = "probability_distribution" ]; then
        test_probability_distribution
    # elif [ $1 = "elliptic_fitting_by_fns" ]; then
        # test_elliptic_fitting_by_fns
    else
        echo "Argument is wrong"
        exit 1
    fi

else
    test_probability_distribution
fi
