#!/usr/bin/python3

from math import sqrt


def gl2lens(a_1, b_1, m_1, q_1):
    n_1 = sqrt(a_1 ** 2 / b_1 ** 2)
    xi_1 = 1.0 / (sqrt(2 * m_1 * abs(a_1)))
    lambda_1 = sqrt(m_1 / (q_1 ** 2 * n_1 ** 2))
    Hc_t = (a_1 ** 2 / b_1) ** (1 / 2)

    return (n_1, xi_1, lambda_1, Hc_t)


def mgl2lens(a_1, b_1, m_1, q_1, a_2, b_2, m_2, q_2):
    n_1 = sqrt(a_1 ** 2 / b_1 ** 2)
    xi_1 = 1.0 / sqrt(2 * m_1 * abs(a_1))
    lambda_1 = sqrt(m_1 / (q_1 ** 2 * n_1 ** 2))

    n_2 = sqrt(a_2 ** 2 / b_2 ** 2)
    xi_2 = 1.0 / sqrt(2 * m_2 * abs(a_2))
    lambda_2 = sqrt(m_2 / (q_2 ** 2 * n_2 ** 2))

    lambda_t = (lambda_1 ** (-2) + lambda_2 ** (-2)) ** (-1/2)
    Hc_t = sqrt(a_1 ** 2 / b_1 + a_2 ** 2 / b_2)

    return (n_1, xi_1, n_2, xi_2, lambda_t, Hc_t)


#########
a_1 = -0.5
b_1 = 0.5
m_1 = 4
q_1 = -1.0

a_2 = -0.5
b_2 = 0.5
m_2 = 8
q_2 = -1.0

n_1, xi_1, n_2, xi_2, lambda_t, Hc_t = mgl2lens(a_1, b_1, m_1, q_1, a_2, b_2, m_2, q_2)

stri = (
    f"""n_1       = {n_1}          \n"""
    f"""xi_1      = {xi_1}         \n"""
    f"""                           \n"""
    f"""n_2       = {n_2}          \n"""
    f"""xi_2      = {xi_2}         \n"""
    f"""                           \n"""
    f"""lambda     = {lambda_t}    \n"""
    f"""H_c        = {Hc_t}        \n"""
)

print(stri)

if xi_1 > lambda_t and xi_2 > lambda_t:
    print("TYPE 1")

elif min(xi_1, xi_2) < lambda_t and max(xi_1, xi_2) > lambda_t:
    print("TYPE 1.5")

else:
    print("TYPE 2")
