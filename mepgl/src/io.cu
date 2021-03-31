#include <string>
#include <experimental/filesystem>

#include "real.cuh"
#include "config.cuh"
#include "cnpy.h"

namespace fs = std::experimental::filesystem;

void loadInput(std::string simulation_name, unsigned int F, unsigned int N, real2 *r, int* comp_domain, int* sc_domain, 
    real *a_1, real *b_1, real *m_xx_1, real *m_yy_1, real *a_2, real *b_2, real *m_xx_2, real *m_yy_2, real *h,
    real *q_1, real *q_2, real *eta, real *gamma, real *delta,
    real2 *psi_1, real2 *psi_2, real2 *a){

    std::string simulation_init_path = "./simulations/"+simulation_name+"/input_data/";

    // Load meshgrids    
    cnpy::NpyArray x_arr = cnpy::npy_load(simulation_init_path + "x.npy");
    cnpy::NpyArray y_arr = cnpy::npy_load(simulation_init_path + "y.npy");
    real *x = x_arr.data<real>();
    real *y = y_arr.data<real>();

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            r[i+N*j].x = (real)x[j+N*i];
            r[i+N*j].y = (real)y[j+N*i];
        }
    }

    // Load superconducting domain and external field  

    cnpy::NpyArray comp_arr = cnpy::npy_load(simulation_init_path + "comp_domain.npy");
    cnpy::NpyArray sc_arr   = cnpy::npy_load(simulation_init_path + "sc_domain.npy");
    int *comp_temp = comp_arr.data<int>();
    int *sc_temp = sc_arr.data<int>();
    
    cnpy::NpyArray a_1_arr = cnpy::npy_load(simulation_init_path + "a_1.npy");
    cnpy::NpyArray b_1_arr = cnpy::npy_load(simulation_init_path + "b_1.npy");
    cnpy::NpyArray m_xx_1_arr = cnpy::npy_load(simulation_init_path + "m_xx_1.npy");
    cnpy::NpyArray m_yy_1_arr = cnpy::npy_load(simulation_init_path + "m_yy_1.npy");
    real *a_1_temp = a_1_arr.data<real>();
    real *b_1_temp = b_1_arr.data<real>();
    real *m_xx_1_temp = m_xx_1_arr.data<real>();
    real *m_yy_1_temp = m_yy_1_arr.data<real>();

    cnpy::NpyArray a_2_arr = cnpy::npy_load(simulation_init_path + "a_2.npy");
    cnpy::NpyArray b_2_arr = cnpy::npy_load(simulation_init_path + "b_2.npy");
    cnpy::NpyArray m_xx_2_arr = cnpy::npy_load(simulation_init_path + "m_xx_2.npy");
    cnpy::NpyArray m_yy_2_arr = cnpy::npy_load(simulation_init_path + "m_yy_2.npy");
    real *a_2_temp = a_2_arr.data<real>();
    real *b_2_temp = b_2_arr.data<real>();
    real *m_xx_2_temp = m_xx_2_arr.data<real>();
    real *m_yy_2_temp = m_yy_2_arr.data<real>();

    cnpy::NpyArray h_arr = cnpy::npy_load(simulation_init_path + "h.npy");
    real *h_temp = h_arr.data<real>();
    
    cnpy::NpyArray q_arr = cnpy::npy_load(simulation_init_path + "q.npy");
    real *q_temp = q_arr.data<real>();
    
    *q_1   = q_temp[0];
    *q_2   = q_temp[1];
    *eta   = q_temp[2];
    *gamma = q_temp[3];
    *delta = q_temp[4];

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            comp_domain[i+N*j] = comp_temp[j+N*i];
            sc_domain[i+N*j] = sc_temp[j+N*i];
        
            a_1[i+N*j] = (real)a_1_temp[j+N*i];
            b_1[i+N*j] = (real)b_1_temp[j+N*i];
            m_xx_1[i+N*j] = (real)m_xx_1_temp[j+N*i];
            m_yy_1[i+N*j] = (real)m_yy_1_temp[j+N*i];
        
            a_2[i+N*j] = (real)a_2_temp[j+N*i];
            b_2[i+N*j] = (real)b_2_temp[j+N*i];
            m_xx_2[i+N*j] = (real)m_xx_2_temp[j+N*i];
            m_yy_2[i+N*j] = (real)m_yy_2_temp[j+N*i];
        
            h[i+N*j] = (real)h_temp[j+N*i];
        }
    }


    // Load frames
    for (int n = 0; n < F; n++)
    {
        cnpy::NpyArray ax_arr = cnpy::npy_load(simulation_init_path + std::to_string(n) + "/" + "ax.npy");
        cnpy::NpyArray ay_arr = cnpy::npy_load(simulation_init_path + std::to_string(n) + "/" + "ay.npy");
        cnpy::NpyArray  u_1_arr = cnpy::npy_load(simulation_init_path + std::to_string(n) + "/" + "u_1.npy");
        cnpy::NpyArray  v_1_arr = cnpy::npy_load(simulation_init_path + std::to_string(n) + "/" + "v_1.npy");
        real *ax_temp = ax_arr.data<real>();
        real *ay_temp = ay_arr.data<real>();
        real *u_1_temp  = u_1_arr.data<real>();
        real *v_1_temp  = v_1_arr.data<real>();

        #ifdef MULTICOMPONENT
        cnpy::NpyArray  u_2_arr = cnpy::npy_load(simulation_init_path + std::to_string(n) + "/" + "u_2.npy");
        cnpy::NpyArray  v_2_arr = cnpy::npy_load(simulation_init_path + std::to_string(n) + "/" + "v_2.npy");
        real *u_2_temp  = u_2_arr.data<real>();
        real *v_2_temp  = v_2_arr.data<real>();
        #endif

        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                a[i+N*j+N*N*n].x = (real)ax_temp[j+N*i];
                a[i+N*j+N*N*n].y = (real)ay_temp[j+N*i];
                psi_1[i+N*j+N*N*n].x = (real)u_1_temp[j+N*i];
                psi_1[i+N*j+N*N*n].y = (real)v_1_temp[j+N*i];

                #ifdef MULTICOMPONENT
                psi_2[i+N*j+N*N*n].x = (real)u_2_temp[j+N*i];
                psi_2[i+N*j+N*N*n].y = (real)v_2_temp[j+N*i];
                #endif
            }
        }
    }
}

void saveOutput(std::string simulation_name, unsigned int F, unsigned int N, real2 *r, int* comp_domain, int* sc_domain, 
    real *a_1, real *b_1, real *m_xx_1, real *m_yy_1, real *a_2, real *b_2, real *m_xx_2, real *m_yy_2, real *h,
    real *q_1, real *q_2, real *eta, real *gamma, real *delta,
    real2 *psi_1, real2 *psi_2, real2 *a,
    real2 *j_1, real2 *j_2, real *b,
    real *fenergy, real *s, real *s_jb){

    std::string simulation_output_path = "./simulations/"+simulation_name+"/output_data/";
    fs::create_directories(simulation_output_path);

    // Save free energy and s
    cnpy::npy_save(simulation_output_path  + "fenergy.npy", &fenergy[0], {F}, "w");
    cnpy::npy_save(simulation_output_path  + "s.npy", &s[0], {F}, "w");
    cnpy::npy_save(simulation_output_path  + "s_jb.npy", &s_jb[0], {F}, "w");

    // Save couplings
    real* q = new real[5];
    q[0] = *q_1;
    q[1] = *q_2;
    q[2] = *eta;
    q[3] = *gamma;
    q[4] = *delta;
    cnpy::npy_save(simulation_output_path  + "q.npy", &q[0], {5}, "w");
    
    // Allocate buffers
    size_t size = N*N;
    std::vector<int> buff_sc(size);
    std::vector<int> buff_comp(size);
    std::vector<real> buff1(size); 
    std::vector<real> buff2(size); 
    std::vector<real> buff3(size); 
    std::vector<real> buff4(size); 

    // Save meshgrids and domains
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            buff_comp[j+N*i] = comp_domain[i+N*j];
            buff_sc[j+N*i] = sc_domain[i+N*j];
            buff1[j+N*i] = (real)r[i+N*j].x;
            buff2[j+N*i] = (real)r[i+N*j].y;
        }
    }

    cnpy::npy_save(simulation_output_path + "comp_domain.npy", &buff_comp[0], {N,N}, "w");
    cnpy::npy_save(simulation_output_path + "sc_domain.npy", &buff_sc[0], {N,N}, "w");
    cnpy::npy_save(simulation_output_path + "x.npy", &buff1[0], {N,N}, "w");
    cnpy::npy_save(simulation_output_path + "y.npy", &buff2[0], {N,N}, "w");

    // Save params part 1
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            buff1[j+N*i] = (real)a_1[i+N*j];
            buff2[j+N*i] = (real)b_1[i+N*j];
            buff3[j+N*i] = (real)m_xx_1[i+N*j];
            buff4[j+N*i] = (real)m_yy_1[i+N*j];
        }
    }

    cnpy::npy_save(simulation_output_path + "a_1.npy", &buff1[0], {N,N}, "w");
    cnpy::npy_save(simulation_output_path + "b_1.npy", &buff2[0], {N,N}, "w");
    cnpy::npy_save(simulation_output_path + "m_xx_1.npy", &buff3[0], {N,N}, "w");
    cnpy::npy_save(simulation_output_path + "m_yy_1.npy", &buff4[0], {N,N}, "w");

    // Save params part 2
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            buff1[j+N*i] = (real)a_2[i+N*j];
            buff2[j+N*i] = (real)b_2[i+N*j];
            buff3[j+N*i] = (real)m_xx_2[i+N*j];
            buff4[j+N*i] = (real)m_yy_2[i+N*j];
        }
    }

    cnpy::npy_save(simulation_output_path + "a_2.npy", &buff1[0], {N,N}, "w");
    cnpy::npy_save(simulation_output_path + "b_2.npy", &buff2[0], {N,N}, "w");
    cnpy::npy_save(simulation_output_path + "m_xx_2.npy", &buff3[0], {N,N}, "w");
    cnpy::npy_save(simulation_output_path + "m_yy_2.npy", &buff4[0], {N,N}, "w");

    // Save external field
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            buff1[j+N*i] = (real)h[i+N*j];
        }
    }

    cnpy::npy_save(simulation_output_path + "h.npy", &buff1[0], {N,N}, "w");

    // Save fields
    for (int n = 0; n < F; n++)
    {
        fs::create_directories(simulation_output_path + "/" + std::to_string(n) + "/" );

        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                buff1[j+N*i] = (real)psi_1[i+N*j+N*N*n].x;
                buff2[j+N*i] = (real)psi_1[i+N*j+N*N*n].y;
                buff3[j+N*i] = (real)a[i+N*j+N*N*n].x;
                buff4[j+N*i] = (real)a[i+N*j+N*N*n].y;
            }
        }
        
        cnpy::npy_save(simulation_output_path + std::to_string(n) + "/" + "u_1.npy", &buff1[0], {N,N}, "w");
        cnpy::npy_save(simulation_output_path + std::to_string(n) + "/" + "v_1.npy", &buff2[0], {N,N}, "w");
        cnpy::npy_save(simulation_output_path + std::to_string(n) + "/" + "ax.npy", &buff3[0], {N,N}, "w");
        cnpy::npy_save(simulation_output_path + std::to_string(n) + "/" + "ay.npy", &buff4[0], {N,N}, "w");

        #ifdef MULTICOMPONENT
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                buff1[j+N*i] = (real)psi_2[i+N*j+N*N*n].x;
                buff2[j+N*i] = (real)psi_2[i+N*j+N*N*n].y;
            }
        }

        cnpy::npy_save(simulation_output_path + std::to_string(n) + "/" + "u_2.npy", &buff1[0], {N,N}, "w");
        cnpy::npy_save(simulation_output_path + std::to_string(n) + "/" + "v_2.npy", &buff2[0], {N,N}, "w");
        #endif
    }

    for (int n = 0; n < F; n++)
    {
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                buff1[j+N*i] = (real)j_1[i+N*j+N*N*n].x;
                buff2[j+N*i] = (real)j_1[i+N*j+N*N*n].y;
                buff3[j+N*i] = (real)b[i+N*j+N*N*n];
            }
        }
        
        cnpy::npy_save(simulation_output_path + std::to_string(n) + "/" + "jx_1.npy", &buff1[0], {N,N}, "w");
        cnpy::npy_save(simulation_output_path + std::to_string(n) + "/" + "jy_1.npy", &buff2[0], {N,N}, "w");
        cnpy::npy_save(simulation_output_path + std::to_string(n) + "/" + "b.npy", &buff3[0], {N,N}, "w");
    }

    #ifdef MULTICOMPONENT
    for (int n = 0; n < F; n++)
    {
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                buff1[j+N*i] = (real)j_2[i+N*j+N*N*n].x;
                buff2[j+N*i] = (real)j_2[i+N*j+N*N*n].y;
            }
        }
        cnpy::npy_save(simulation_output_path + std::to_string(n) + "/" + "jx_2.npy", &buff1[0], {N,N}, "w");
        cnpy::npy_save(simulation_output_path + std::to_string(n) + "/" + "jy_2.npy", &buff2[0], {N,N}, "w");
    }
    #endif
}


void saveStats(std::string simulation_name, unsigned int F, unsigned int itermax, real *fenergy, real *g_norm2){

    std::string simulation_output_path = "./simulations/"+simulation_name+"/output_data/";
    fs::create_directories(simulation_output_path);

    std::vector<real> fenergy_buff(itermax*F); 
    std::vector<real> g_norm2_buff(itermax*F); 

    for (int n = 0; n < F; n++)
    {
        for (int k = 0; k < itermax; k++)
        {

            fenergy_buff[n+F*k] = fenergy[k+itermax*n];
            g_norm2_buff[n+F*k] = g_norm2[k+itermax*n];
        }   
    }
        cnpy::npy_save(simulation_output_path  + "-energy.npy", &fenergy[0], {F, itermax}, "w");
        cnpy::npy_save(simulation_output_path  + "-g_norm2.npy", &g_norm2[0], {F, itermax}, "w");
}