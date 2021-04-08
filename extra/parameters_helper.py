#!/usr/bin/python3

from math import sqrt

q_1    = -1.0
a_1    = -1.0
b_1    = 1.0
m_xx_1 = 1.0
m_yy_1 = 1.0

m_1 = (m_xx_1 + m_yy_1)/2.0

q_2    = -1.0
a_2    = -1.0
b_2    = 1.0
m_xx_2 = 2.5
m_yy_2 = 2.5

m_2 = (m_xx_2 + m_yy_2)/2.0

##################################################

hc_1 = sqrt(a_1**2/b_1)
xi_x_1 = 1.0/(2*sqrt(m_xx_1*abs(a_1)))
xi_y_1 = 1.0/(2*sqrt(m_yy_1*abs(a_1)))
xi_1   = 1.0/(2*sqrt(m_1*abs(a_1)))
lambda_x_1 = sqrt(m_xx_1)/(abs(q_1)*sqrt(abs(a_1)/b_1))
lambda_y_1 = sqrt(m_yy_1)/(abs(q_1)*sqrt(abs(a_1)/b_1))
kappa_x_1 = lambda_x_1/xi_x_1
kappa_y_1 = lambda_y_1/xi_y_1

lambda_1 = sqrt(m_1)/(abs(q_1)*sqrt(abs(a_1)/b_1))

hc_2 = sqrt(a_2**2/b_2)
xi_x_2 = 1.0/(2*sqrt(m_xx_2*abs(a_2)))
xi_y_2 = 1.0/(2*sqrt(m_yy_2*abs(a_2)))
xi_2 = 1.0/(2*sqrt(m_2*abs(a_2)))

lambda_x_2 = sqrt(m_xx_2)/(abs(q_2)*sqrt(abs(a_2)/b_2))
lambda_y_2 = sqrt(m_yy_2)/(abs(q_2)*sqrt(abs(a_2)/b_2))
kappa_x_2 = lambda_x_2/xi_x_2
kappa_y_2 = lambda_y_2/xi_y_2

lambda_2 = sqrt(m_2)/(abs(q_2)*sqrt(abs(a_2)/b_2))

lambda_t = sqrt((lambda_1**2 * lambda_2**2)/(lambda_1**2 + lambda_2**2))
kappa_1 = lambda_t/xi_1
kappa_2 = lambda_t/xi_2

stri =( f""" hc_1       = {hc_1}           \n"""
        f""" xi_x_1     = {xi_x_1}         \n"""
        f""" xi_y_1     = {xi_y_1}         \n"""
        f""" lambda_x_1 = {lambda_x_1}     \n"""
        f""" lambda_y_1 = {lambda_y_1}     \n"""
        f""" kappa_x_1  = {kappa_x_1}      \n"""
        f""" kappa_y_1  = {kappa_y_1}      \n"""
        f"""                               \n"""
        f""" hc_2       = {hc_2}           \n"""
        f""" xi_x_2     = {xi_x_2}         \n"""
        f""" xi_y_2     = {xi_y_2}         \n"""
        f""" lambda_x_2 = {lambda_x_2}     \n"""
        f""" lambda_y_2 = {lambda_y_2}     \n"""
        f""" kappa_x_2  = {kappa_x_2}      \n"""
        f""" kappa_y_2  = {kappa_y_2}      \n"""
        f"""                               \n"""
        f""" lambda      = {lambda_t}      \n"""
        f""" kappa_1     = {kappa_1}       \n"""
        f""" kappa_2     = {kappa_2}       \n"""
       
       )


print(stri)


hc_1 = 1
xi_1 = 2

b_1 = 1
a_1 = hc_1 * sqrt(b_1)
m_1 = 1/(4*xi_1**2*a_1)


xi_1 = 2
xi_2 = 0.5
lambda_t = 1


