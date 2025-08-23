**LinsPlit: An O(n) Trend Change Point Detection Algorithm.**

This repository contains the implementation of the LinsPlit algorithm described in the accompanying technical report.The method finds trend changepoints using least squares estimators applied to piecewise linear regression.

The central approach derives analytical expressions for slope estimators, analyzes their asymptotic behavior, and obtains measures of interest under specific assumptions that enable closed-form results. While initially designed for single change point detection, the method has been extended to handle multiple change points with maintained numerical efficiency.

The implementation follows the theoretical framework presented in the paper, including the derivation of β_left(t) and β_right(t) expressions and the O(n) computational approach.

The repository includes all code used to generate the results in the technical report.

