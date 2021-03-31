/******************************************************
 * Simplified String method for Ginzburg Landau System 
 ******************************************************/

#include <future>
#include <thread>
#include <chrono>
#include <string>
#include <iostream>
#include <fstream>
#include <cstdio>
#include <cmath>

#include <unistd.h>
#include <fcntl.h>

#include <boost/program_options.hpp>

#include <cusparse_v2.h>

#include "cuda_errors.cuh"

#include "real.cuh"
#include "parallel_sum.cuh"
#include "nlcg.cuh"
#include "interpolation.cuh"
#include "io.cuh"

#include "config.cuh"

// Choice of the metric. Option: s or s_jb
#define S s_jb

namespace po = boost::program_options;

static cusparseHandle_t handle;

__host__
void loop( bool &stop_flag,
           unsigned int iterations, int &relaxation_steps, unsigned int F, 
           int* dev_comp_nn_map, int* dev_sc_nn_map, 
           real* dev_a_1, real* dev_b_1, real* dev_m_xx_1, real* dev_m_yy_1,
           real* dev_a_2, real* dev_b_2, real* dev_m_xx_2, real* dev_m_yy_2,
           real* dev_h, real* dev_q_1, real* dev_q_2, real *dev_eta, real *dev_gamma, real *dev_delta,
           real2* &dev_a, real2* &dev_psi_1, real2* &dev_psi_2,
           real* dev_b, real2 *dev_j_1, real2 *dev_j_2,
           real* fenergy, real *s, real *s_jb,
           bool &nlcg, char &mode){

    // Stuff for linesearch
    real4 *poly_coeff = nullptr;
    real *alpha = nullptr;
    checkCudaErrors(cudaMallocManaged(&poly_coeff, F*sizeof(real4)));
    checkCudaErrors(cudaMallocManaged(&alpha, F*sizeof(real)));

    // Norm squared of gradient
    real *g_norm2 = nullptr;
    real *g_norm2_old = nullptr;    
    real *g_delta_g = nullptr;    
    checkCudaErrors(cudaMallocManaged(&g_norm2, F*sizeof(real)));
    checkCudaErrors(cudaMallocManaged(&g_norm2_old, F*sizeof(real)));
    checkCudaErrors(cudaMallocManaged(&g_delta_g, F*sizeof(real)));
        
    // Buffer for interpolation
    real2 *dev_a_new = nullptr; 
    real2 *dev_psi_1_new = nullptr;
    real2 *dev_psi_2_new = nullptr;
    checkCudaErrors(cudaMalloc(&dev_a_new, F*N*N*sizeof(real2)));
    checkCudaErrors(cudaMalloc(&dev_psi_1_new, F*N*N*sizeof(real2)));
    checkCudaErrors(cudaMalloc(&dev_psi_2_new, F*N*N*sizeof(real2)));

    // Gradient
    real2 *dev_a_g =  nullptr;
    real2 *dev_psi_1_g =  nullptr;
    real2 *dev_psi_2_g =  nullptr;
    checkCudaErrors(cudaMalloc(&dev_a_g, F*N*N*sizeof(real2)));
    checkCudaErrors(cudaMalloc(&dev_psi_1_g, F*N*N*sizeof(real2)));
    checkCudaErrors(cudaMalloc(&dev_psi_2_g, F*N*N*sizeof(real2)));

    real2 *dev_a_g_old =  nullptr;
    real2 *dev_psi_1_g_old =  nullptr;
    real2 *dev_psi_2_g_old =  nullptr;
    checkCudaErrors(cudaMalloc(&dev_a_g_old, F*N*N*sizeof(real2)));
    checkCudaErrors(cudaMalloc(&dev_psi_1_g_old, F*N*N*sizeof(real2)));
    checkCudaErrors(cudaMalloc(&dev_psi_2_g_old, F*N*N*sizeof(real2)));
    
    real2 *dev_a_d =  nullptr;
    real2 *dev_psi_1_d =  nullptr;
    real2 *dev_psi_2_d =  nullptr;
    checkCudaErrors(cudaMalloc(&dev_a_d, F*N*N*sizeof(real2)));
    checkCudaErrors(cudaMalloc(&dev_psi_1_d, F*N*N*sizeof(real2)));
    checkCudaErrors(cudaMalloc(&dev_psi_2_d, F*N*N*sizeof(real2)));

    real *dev_g_norm2_density = nullptr;
    real *dev_g_delta_g_density = nullptr;
    checkCudaErrors(cudaMalloc(&dev_g_norm2_density,F*N*N*sizeof(real)));
    checkCudaErrors(cudaMalloc(&dev_g_delta_g_density,F*N*N*sizeof(real)));

    real4 *dev_poly_coeff_density = nullptr;
    real  *dev_fenergy_density = nullptr;    
    checkCudaErrors(cudaMalloc(&dev_poly_coeff_density, F*N*N*sizeof(real4)));
    checkCudaErrors(cudaMalloc(&dev_fenergy_density, F*N*N*sizeof(real)));

    real *dev_prime_norm2_buffer = nullptr;
    real *dev_g_dot_prime_buffer = nullptr;
    checkCudaErrors(cudaMalloc(&dev_prime_norm2_buffer, F*N*N*sizeof(real)));
    checkCudaErrors(cudaMalloc(&dev_g_dot_prime_buffer, F*N*N*sizeof(real)));

    real *prime_norm2 = nullptr;
    real *g_dot_prime = nullptr;
    checkCudaErrors(cudaMallocManaged(&prime_norm2, F*sizeof(real)));
    checkCudaErrors(cudaMallocManaged(&g_dot_prime, F*sizeof(real)));

    real *dev_psi_abs_1 = nullptr;
    real *dev_psi_abs_2 = nullptr;
    checkCudaErrors(cudaMalloc(&dev_psi_abs_1, F*N*N*sizeof(real)));
    checkCudaErrors(cudaMalloc(&dev_psi_abs_2, F*N*N*sizeof(real)));

    real2 *dev_cd_1 = nullptr;
    real2 *dev_cd_2 = nullptr;
    checkCudaErrors(cudaMalloc(&dev_cd_1, F*N*N*sizeof(real2)));
    checkCudaErrors(cudaMalloc(&dev_cd_2, F*N*N*sizeof(real2)));
   
    /****************************** INITIALIZATION *********************************/

    // Initialize to zero gradient and direction
    checkCudaErrors(cudaMemset(dev_a_g, 0, F*N*N*sizeof(real2))); 	
    checkCudaErrors(cudaMemset(dev_psi_1_g, 0, F*N*N*sizeof(real2))); 	
    checkCudaErrors(cudaMemset(dev_psi_2_g, 0, F*N*N*sizeof(real2))); 	
    checkCudaErrors(cudaMemset(dev_a_g_old, 0, F*N*N*sizeof(real2))); 	
    checkCudaErrors(cudaMemset(dev_psi_1_g_old, 0, F*N*N*sizeof(real2))); 	
    checkCudaErrors(cudaMemset(dev_psi_2_g_old, 0, F*N*N*sizeof(real2)));	
    checkCudaErrors(cudaMemset(dev_a_d, 0, F*N*N*sizeof(real2))); 	  	
    checkCudaErrors(cudaMemset(dev_psi_1_d, 0, F*N*N*sizeof(real2)));  	
    checkCudaErrors(cudaMemset(dev_psi_2_d, 0, F*N*N*sizeof(real2))); 

    // real *fenergy_at_step = new real[F*iterations];
    // real *g_norm2_at_step = new real[F*iterations];    
    
    real *fenergy_old = new real[F];

    for(int n = 0; n < F; n++){
        g_norm2[n] = INFINITY;
        g_norm2_old[n] = INFINITY;
        fenergy_old[n] = INFINITY;
    }

        for(int counter = 0; counter < iterations; counter ++){

        printf( "\n" );
        printf( "Iteration:   %d\n", counter + 1);
            
        if(mode=='S'){

            printf("=> Relaxation steps: ");             
            nlcgSteps(
                relaxation_steps, F,
                dev_comp_nn_map, dev_sc_nn_map, 
                dev_a_1, dev_b_1, dev_m_xx_1, dev_m_yy_1, 
                dev_a_2, dev_b_2, dev_m_xx_2, dev_m_yy_2, 
                dev_h, dev_q_1, dev_q_2, dev_eta, dev_gamma, dev_delta,
                dev_a, dev_psi_1, dev_psi_2, 
                dev_a_g, dev_psi_1_g, dev_psi_2_g, 
                dev_a_g_old, dev_psi_1_g_old, dev_psi_2_g_old, 
                dev_a_d, dev_psi_1_d, dev_psi_2_d,
                dev_g_norm2_density, dev_g_delta_g_density,
                g_norm2, g_norm2_old, g_delta_g, alpha,
                dev_fenergy_density, dev_poly_coeff_density, fenergy, poly_coeff,
                nlcg, false
                );
        } else if(mode=='M'){

            printf("=> Relaxation steps: ");
            nlcgSteps(
                relaxation_steps, F,
                dev_comp_nn_map, dev_sc_nn_map, 
                dev_a_1, dev_b_1, dev_m_xx_1, dev_m_yy_1, 
                dev_a_2, dev_b_2, dev_m_xx_2, dev_m_yy_2, 
                dev_h, dev_q_1, dev_q_2, dev_eta, dev_gamma, dev_delta,
                dev_a, dev_psi_1, dev_psi_2, 
                dev_a_g, dev_psi_1_g, dev_psi_2_g, 
                dev_a_g_old, dev_psi_1_g_old, dev_psi_2_g_old, 
                dev_a_d, dev_psi_1_d, dev_psi_2_d,
                dev_g_norm2_density, dev_g_delta_g_density,
                g_norm2, g_norm2_old, g_delta_g, alpha,
                dev_fenergy_density, dev_poly_coeff_density, fenergy, poly_coeff,
                nlcg, true
                );
        } else if(mode=='L'){


            printf("=> Linear interpolation step \n");
            arcLength(N, F, s, dev_a, dev_psi_1, dev_psi_2);

            computeJB(F, dev_comp_nn_map, dev_sc_nn_map,
                dev_a_1, dev_b_1, dev_m_xx_1, dev_m_yy_1, 
                dev_a_2, dev_b_2, dev_m_xx_2, dev_m_yy_2, 
                dev_h, dev_q_1, dev_q_2,
                dev_a, dev_psi_1, dev_psi_2,
                dev_b, dev_j_1, dev_j_2);
            arcLengthJB(N, F, s_jb, dev_b, dev_j_1, dev_j_2);
            
            // computeAbsGrad(F, dev_sc_nn_map,
            //                dev_q_1, dev_q_2,
            //                dev_a, dev_psi_1, dev_psi_2,
            //                dev_psi_abs_1, dev_cd_1,
            //                dev_psi_abs_2, dev_cd_2);
            // arcLengthAbsGrad(N, F, s_jb, dev_psi_abs_1, dev_cd_1,
            //                              dev_psi_abs_1, dev_cd_2);

            redistributeLinear(handle, N, F, S, dev_a, dev_psi_1, dev_psi_2, 
                                         F, dev_a_new, dev_psi_1_new, dev_psi_2_new);
    
            // checkCudaErrors(cudaMemcpy(dev_a_new, dev_a, F*N*N*sizeof(real2), cudaMemcpyDeviceToDevice));
            // checkCudaErrors(cudaMemcpy(dev_psi_1_new, dev_psi_1, F*N*N*sizeof(real2), cudaMemcpyDeviceToDevice));
            // checkCudaErrors(cudaMemcpy(dev_psi_2_new, dev_psi_2, F*N*N*sizeof(real2), cudaMemcpyDeviceToDevice));

            real2* temp_a = dev_a;
            dev_a = dev_a_new;
            dev_a_new = temp_a;
    
            real2* temp_psi_1 = dev_psi_1;
            dev_psi_1 = dev_psi_1_new;
            dev_psi_1_new = temp_psi_1;
    
            real2* temp_psi_2 = dev_psi_2;
            dev_psi_2 = dev_psi_2_new;
            dev_psi_2_new = temp_psi_2;

            printf("=> Relaxation steps: ");
            for(int n = 1; n < F-1; n++){
                g_norm2[n] = INFINITY;
                g_norm2_old[n] = INFINITY;
            }

            nlcgSteps(
                relaxation_steps, F,
                dev_comp_nn_map, dev_sc_nn_map, 
                dev_a_1, dev_b_1, dev_m_xx_1, dev_m_yy_1, 
                dev_a_2, dev_b_2, dev_m_xx_2, dev_m_yy_2, 
                dev_h, dev_q_1, dev_q_2, dev_eta, dev_gamma, dev_delta,
                dev_a, dev_psi_1, dev_psi_2, 
                dev_a_g, dev_psi_1_g, dev_psi_2_g, 
                dev_a_g_old, dev_psi_1_g_old, dev_psi_2_g_old, 
                dev_a_d, dev_psi_1_d, dev_psi_2_d,
                dev_g_norm2_density, dev_g_delta_g_density,
                g_norm2, g_norm2_old, g_delta_g, alpha,
                dev_fenergy_density, dev_poly_coeff_density, fenergy, poly_coeff,
                nlcg, false
                );


        } else if(mode=='C'){

            #ifndef DOUBLE_PRECISION

            printf("=> Cubic interpolation step \n");
            arcLength(N, F, s, dev_a, dev_psi_1, dev_psi_2);          
            computeJB(F, dev_comp_nn_map, dev_sc_nn_map,
                dev_a_1, dev_b_1, dev_m_xx_1, dev_m_yy_1, 
                dev_a_2, dev_b_2, dev_m_xx_2, dev_m_yy_2, 
                dev_h, dev_q_1, dev_q_2, 
                dev_a, dev_psi_1, dev_psi_2,
                dev_b, dev_j_1, dev_j_2);
            arcLengthJB(N, F, s_jb, dev_b, dev_j_1, dev_j_2);
            redistributeCubic(handle, N, F, S, dev_a, dev_psi_1, dev_psi_2, 
                                                F, dev_a_new, dev_psi_1_new, dev_psi_2_new);
    
            real2* temp_a = dev_a;
            dev_a = dev_a_new;
            dev_a_new = temp_a;
    
            real2* temp_psi_1 = dev_psi_1;
            dev_psi_1 = dev_psi_1_new;
            dev_psi_1_new = temp_psi_1;
    
            real2* temp_psi_2 = dev_psi_2;
            dev_psi_2 = dev_psi_2_new;
            dev_psi_2_new = temp_psi_2;

            for(int n = 1; n < F-1; n++){
                g_norm2[n] = INFINITY;
                g_norm2_old[n] = INFINITY;
            }

            nlcgSteps(
                relaxation_steps, F,
                dev_comp_nn_map, dev_sc_nn_map, 
                dev_a_1, dev_b_1, dev_m_xx_1, dev_m_yy_1, 
                dev_a_2, dev_b_2, dev_m_xx_2, dev_m_yy_2, 
                dev_h, dev_q_1, dev_q_2, dev_eta, dev_gamma, dev_delta,
                dev_a, dev_psi_1, dev_psi_2, 
                dev_a_g, dev_psi_1_g, dev_psi_2_g, 
                dev_a_g_old, dev_psi_1_g_old, dev_psi_2_g_old, 
                dev_a_d, dev_psi_1_d, dev_psi_2_d,
                dev_g_norm2_density, dev_g_delta_g_density,
                g_norm2, g_norm2_old, g_delta_g, alpha,
                dev_fenergy_density, dev_poly_coeff_density, fenergy, poly_coeff,
                nlcg, false
                );

                #endif
        }
                

        /**************************** PRINTING AND SAVING *****************/

        // Computing mean norm
        real g_norm_mean = 0;

        for(int n = 0; n<F; n++){
            g_norm_mean += sqrt(g_norm2[n]);
        } 

        g_norm_mean = g_norm_mean/(F*N*N);
        
        // computeFreeEnergy(F,
        //     dev_comp_nn_map, dev_sc_nn_map, 
        //     dev_a_1, dev_b_1, dev_m_xx_1, dev_m_yy_1, 
        //     dev_a_2, dev_b_2, dev_m_xx_2, dev_m_yy_2, 
        //     dev_h, dev_q_1, dev_q_2, dev_eta,
        //     dev_a, dev_psi_1, dev_psi_2, fenergy);
        
        // Print stuff
        printf("\n");
        printf("┌──────┬──────┬─────────────────┬─────────────────┬────────────────┬────────────────┐\n");
        printf("│ Sys# │   s  │  Free Energy    │  Delta Energy   │      ||g||     │      alpha     │\n");        
        printf("├──────┼──────┼─────────────────┼─────────────────┼────────────────┼────────────────┤\n");
        for(int n = 0; n<F; n+=max((F-1)/10,1)) 
            printf("│  %3.3d │ %4.2f │ %+.8e │ %+.8e │ %.8e │ %.8e │\n", n, s_jb[n], fenergy[n], fenergy[n]-fenergy_old[n],sqrt(g_norm2[n])/(N*N), alpha[n]);
        printf("└──────┴──────┴─────────────────┴─────────────────┴────────────────┴────────────────┘\n");
        
        printf("\n");
        printf("||g||_mean      %e       \n", g_norm_mean);
        printf("\n");
        
        // Save stats
        for(int n = 0; n<F; n++){
            fenergy_old[n] = fenergy[n];
        }

        // Save stats
        // for(int n = 0; n<F; n++){
        //     g_norm2_at_step[counter+n*iterations] = g_norm2[n];
        //     fenergy_at_step[counter+n*iterations] = fenergy[n];
        // }
        
        std::ofstream temp_file;
        temp_file.open ("energy.csv");
        temp_file << std::scientific;
        for(int n = 0; n<F-1; n++){
            temp_file << fenergy[n] << ",";
        }
        temp_file << fenergy[F-1];
        temp_file.close();

        if(stop_flag){
            break;
        }    
    }

    checkCudaErrors(cudaFree(dev_a_new));
    checkCudaErrors(cudaFree(dev_psi_1_new));
    checkCudaErrors(cudaFree(dev_psi_2_new));

    checkCudaErrors(cudaFree(dev_a_g));
    checkCudaErrors(cudaFree(dev_psi_1_g));
    checkCudaErrors(cudaFree(dev_psi_2_g));

    checkCudaErrors(cudaFree(dev_a_g_old));
    checkCudaErrors(cudaFree(dev_psi_1_g_old));
    checkCudaErrors(cudaFree(dev_psi_2_g_old));
    
    checkCudaErrors(cudaFree(dev_a_d));
    checkCudaErrors(cudaFree(dev_psi_1_d));
    checkCudaErrors(cudaFree(dev_psi_2_d));
    
    checkCudaErrors(cudaFree(dev_g_norm2_density));
    checkCudaErrors(cudaFree(dev_g_delta_g_density));

    checkCudaErrors(cudaFree(dev_poly_coeff_density));
    checkCudaErrors(cudaFree(dev_fenergy_density));
    
    checkCudaErrors(cudaFree(g_norm2));
    checkCudaErrors(cudaFree(g_norm2_old));
    checkCudaErrors(cudaFree(poly_coeff));
    checkCudaErrors(cudaFree(alpha));
    
    // delete[] fenergy_at_step;
    // delete[] g_norm2_at_step;  
}


void parseArgs(int argc, const char *argv[], std::string &simname, char &mode, int &F, int &F_out, int &iterations){
    try{
        po::options_description desc{"Options"};
        desc.add_options()
        ("help,h", "Help screen")
        ("simname", po::value<std::string>(&simname), "Simulation name")
        ("mode", po::value<char>(&mode), "Mode selected (M, S, L, C)")
        ("Fin",  po::value<int>(&F), "Number of frames of the input guess")
        ("Fout", po::value<int>(&F_out), "Number of frames of the output")
        ("iterations", po::value<int>(&iterations), "Number of iterations");

        po::variables_map vm;
        po::store(parse_command_line(argc, argv, desc), vm);
        po::notify(vm);

        if (vm.count("help")){
            std::cout << desc << '\n';
            exit(0);
        }

    }
    catch (const po::error &ex){
        std::cerr << ex.what() << '\n';
    }
}


int main(int argc, const char *argv[]){
    std::string simulation_name;
    char mode;
    int F;
    int F_out;
    int iterations;

    //Parse line arguments
    parseArgs(argc, argv, simulation_name, mode, F, F_out, iterations);
    
    // Disabling buffering to printf istantaneouly
    setbuf(stdout, NULL);

    // Set the device
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    printf("Detected %d CUDA Capable device(s)\n", deviceCount);
    cudaSetDevice(devNumber);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, devNumber);

    printf("\nUsing device %d: \"%s\"\n", devNumber, deviceProp.name);

    printf("\nSimulation name: %s\n", &simulation_name[0]);

    // Create cusparse handle
    checkCudaErrors(cusparseCreate(&handle));

    /***********************ALLOCATE MEMORY**************************/

    // Domain
    real2 *r = new real2[N*N];
        
    // Superconducting domain
    int *comp_domain = new int[N*N];
    int *sc_domain = new int[N*N];

    // Parameters
    real *a_1 = new real[N*N];
    real *b_1 = new real[N*N];
    real *m_xx_1 = new real[N*N];
    real *m_yy_1 = new real[N*N];
    
    real *a_2 = new real[N*N];
    real *b_2 = new real[N*N];
    real *m_xx_2 = new real[N*N];
    real *m_yy_2 = new real[N*N];
    
    real *q_1 = new real;
    real *q_2 = new real;
    real *eta = new real;
    real *gamma = new real;
    real *delta = new real;

    // External field
    real *h = new real[N*N];
    
    // Fields
    real2 *psi_1 = new real2[F*N*N];
    real2 *psi_2 = new real2[F*N*N];
    real2 *a = new real2[F*N*N];

    real2 *j_1 = new real2[F*N*N];
    real2 *j_2 = new real2[F*N*N];
    real *b = new real[F*N*N];
    
    // Arc length and fenergy
    real *s = nullptr;
    real *s_jb = nullptr;
    real *fenergy = nullptr;
    
    checkCudaErrors(cudaMallocManaged(&s, F*sizeof(real)));
    checkCudaErrors(cudaMallocManaged(&s_jb, F*sizeof(real)));
    checkCudaErrors(cudaMallocManaged(&fenergy, F*sizeof(real)));

    // Device variables
    
    int *dev_comp_domain = nullptr; 
    int *dev_comp_nn_map = nullptr; 
    checkCudaErrors(cudaMalloc(&dev_comp_domain, N*N*sizeof(int)));
    checkCudaErrors(cudaMalloc(&dev_comp_nn_map, N*N*sizeof(int)));
    
    int *dev_sc_domain = nullptr; 
    int *dev_sc_nn_map = nullptr;
    checkCudaErrors(cudaMalloc(&dev_sc_domain, N*N*sizeof(int)));
    checkCudaErrors(cudaMalloc(&dev_sc_nn_map, N*N*sizeof(int)));

    // Parameters
    real *dev_a_1 = nullptr;
    real *dev_b_1 = nullptr;
    real *dev_m_xx_1 = nullptr;
    real *dev_m_yy_1 = nullptr;
    checkCudaErrors(cudaMalloc(&dev_a_1, N*N*sizeof(real)));
    checkCudaErrors(cudaMalloc(&dev_b_1, N*N*sizeof(real)));
    checkCudaErrors(cudaMalloc(&dev_m_xx_1, N*N*sizeof(real)));
    checkCudaErrors(cudaMalloc(&dev_m_yy_1, N*N*sizeof(real)));

    real *dev_a_2 = nullptr;
    real *dev_b_2 = nullptr;
    real *dev_m_xx_2 = nullptr;
    real *dev_m_yy_2 = nullptr;
    checkCudaErrors(cudaMalloc(&dev_a_2, N*N*sizeof(real)));
    checkCudaErrors(cudaMalloc(&dev_b_2, N*N*sizeof(real)));
    checkCudaErrors(cudaMalloc(&dev_m_xx_2, N*N*sizeof(real)));
    checkCudaErrors(cudaMalloc(&dev_m_yy_2, N*N*sizeof(real)));

    real *dev_q_1 = nullptr;
    real *dev_q_2 = nullptr;
    real *dev_eta = nullptr;
    real *dev_gamma = nullptr;
    real *dev_delta = nullptr;
    checkCudaErrors(cudaMalloc(&dev_q_1, sizeof(real)));
    checkCudaErrors(cudaMalloc(&dev_q_2, sizeof(real)));
    checkCudaErrors(cudaMalloc(&dev_eta, sizeof(real)));
    checkCudaErrors(cudaMalloc(&dev_gamma, sizeof(real)));
    checkCudaErrors(cudaMalloc(&dev_delta, sizeof(real)));

    real *dev_h = nullptr;
    checkCudaErrors(cudaMalloc(&dev_h, N*N*sizeof(real)));

    real2 *dev_psi_1 = nullptr;
    real2 *dev_psi_2 = nullptr;
    real2 *dev_a = nullptr; 

    checkCudaErrors(cudaMalloc(&dev_psi_1, F*N*N*sizeof(real2)));
    checkCudaErrors(cudaMalloc(&dev_psi_2, F*N*N*sizeof(real2)));
    checkCudaErrors(cudaMalloc(&dev_a, F*N*N*sizeof(real2)));
    
    real2 *dev_j_1 = nullptr;
    real2 *dev_j_2 = nullptr;
    real *dev_b = nullptr; 

    checkCudaErrors(cudaMalloc(&dev_j_1, F*N*N*sizeof(real2)));
    checkCudaErrors(cudaMalloc(&dev_j_2, F*N*N*sizeof(real2)));
    checkCudaErrors(cudaMalloc(&dev_b, F*N*N*sizeof(real)));
    
    // Load input data
    printf("Loading input data...  ");
    loadInput(simulation_name, F, N, r, comp_domain, sc_domain, 
               a_1, b_1, m_xx_1, m_yy_1, a_2, b_2, m_xx_2, m_yy_2, h,
               q_1, q_2, eta, gamma, delta,
               psi_1, psi_2, a);
    printf("done.\n\n");

    // Move loaded data to device
    checkCudaErrors(cudaMemcpy(dev_comp_domain, comp_domain, N*N*sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dev_sc_domain, sc_domain, N*N*sizeof(int), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(dev_a_1, a_1, N*N*sizeof(real), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dev_b_1, b_1, N*N*sizeof(real), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dev_m_xx_1, m_xx_1, N*N*sizeof(real), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dev_m_yy_1, m_yy_1, N*N*sizeof(real), cudaMemcpyHostToDevice));

    #ifdef MULTICOMPONENT
    checkCudaErrors(cudaMemcpy(dev_a_2, a_2, N*N*sizeof(real), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dev_b_2, b_2, N*N*sizeof(real), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dev_m_xx_2, m_xx_2, N*N*sizeof(real), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dev_m_yy_2, m_yy_2, N*N*sizeof(real), cudaMemcpyHostToDevice));
    #endif

    checkCudaErrors(cudaMemcpy(dev_q_1, q_1, sizeof(real), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dev_q_2, q_2, sizeof(real), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dev_eta, eta, sizeof(real), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dev_gamma, gamma, sizeof(real), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dev_delta, delta, sizeof(real), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(dev_h, h, N*N*sizeof(real), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dev_psi_1, psi_1, F*N*N*sizeof(real2), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dev_a, a, F*N*N*sizeof(real2), cudaMemcpyHostToDevice));

    #ifdef MULTICOMPONENT
    checkCudaErrors(cudaMemcpy(dev_psi_2, psi_2, F*N*N*sizeof(real2), cudaMemcpyHostToDevice));
    #endif
    checkCudaErrors(cudaDeviceSynchronize());

    // Build NN Map
    buildNearestNeighbourMap<<<gridSizeX, 1024>>>(dev_comp_domain, dev_comp_nn_map);   
    buildNearestNeighbourMap<<<gridSizeX, 1024>>>(dev_sc_domain, dev_sc_nn_map);   
    checkCudaErrors(cudaDeviceSynchronize());

    bool nlcg = true;
    
    computeFreeEnergy(F, dev_comp_nn_map, dev_sc_nn_map, 
                        dev_a_1, dev_b_1, dev_m_xx_1, dev_m_yy_1, 
                        dev_a_2, dev_b_2, dev_m_xx_2, dev_m_yy_2, 
                        dev_h, dev_q_1, dev_q_2, dev_eta, dev_gamma, dev_delta,
                        dev_a, dev_psi_1, dev_psi_2,
                        fenergy);

    computeJB(F, dev_comp_nn_map, dev_sc_nn_map,
                    dev_a_1, dev_b_1, dev_m_xx_1, dev_m_yy_1, 
                    dev_a_2, dev_b_2, dev_m_xx_2, dev_m_yy_2, 
                    dev_h, dev_q_1, dev_q_2,
                    dev_a, dev_psi_1, dev_psi_2,
                    dev_b, dev_j_1, dev_j_2);



    /************************************************** LOOP *****************************************************/

    bool stop_flag = false;
    int relaxation_steps = default_relaxation_step_number;
    char buffer[8];

    printf("┌──────┬────────────────┐ \n");
    printf("| Sys# |  Free Energy   | \n");        
    printf("├──────┼────────────────┤ \n");
    for(int n = 0; n<F; n+=max((F-1)/10,1)) 
        printf("|  %3.3d | %.8e | \n", n, fenergy[n]);
    printf("└──────┴────────────────┘ \n");

    printf("Starting MEP-MGL.\n\n");
 
    fcntl(0, F_SETFL, fcntl(0, F_GETFL) | O_NONBLOCK);
    
    auto check_exit = [&]() -> void{
        cudaSetDevice(devNumber);

        while(true){
            usleep(200);
    
            if(stop_flag){
                break;
            }
    
            int numRead = read(0, buffer, 4);
            if(buffer[0]=='q'){
                printf("I am quitting! Please wait.\n");
                stop_flag = true;
                break;
            }
            
            else if(buffer[0]=='+'){
                relaxation_steps += 1;
                printf("relaxation_steps = %d\n", relaxation_steps);
            }
            
            else if(buffer[0]=='-'){
                if (relaxation_steps>1) relaxation_steps -= 1;
                printf("relaxation_steps = %d\n", relaxation_steps);
            }

            else if(buffer[0]=='C'){
                nlcg = !nlcg;
                printf("NLCG set %s", nlcg ? "ON" : "OFF");
            }

            buffer[0] = '\0';
        }    
    };

    std::thread input_thread(check_exit);

    // Set the device for the child thread
    cudaSetDevice(devNumber);

    loop(stop_flag,
        iterations, relaxation_steps, F, 
        dev_comp_nn_map, dev_sc_nn_map, 
        dev_a_1, dev_b_1, dev_m_xx_1, dev_m_yy_1,
        dev_a_2, dev_b_2, dev_m_xx_2, dev_m_yy_2,
        dev_h, dev_q_1, dev_q_2, dev_eta, dev_gamma, dev_delta,
        dev_a, dev_psi_1, dev_psi_2,
        dev_b, dev_j_1, dev_j_2,
        fenergy, s, s_jb,
        nlcg, mode);
    
    stop_flag = true;

    if(input_thread.joinable()){
        input_thread.join();
        cudaSetDevice(devNumber);
    }

    
    /********************************SAVE DATA AND FREE MEMORY**************************************************/
  
    if (F_out != F){
        arcLength(N, F, s, dev_a, dev_psi_1, dev_psi_2);
        computeJB(F, dev_comp_nn_map, dev_sc_nn_map,
            dev_a_1, dev_b_1, dev_m_xx_1, dev_m_yy_1, 
            dev_a_2, dev_b_2, dev_m_xx_2, dev_m_yy_2, 
            dev_h, dev_q_1, dev_q_2,
            dev_a, dev_psi_1, dev_psi_2,
            dev_b, dev_j_1, dev_j_2);    
        arcLengthJB(N, F, s_jb, dev_b, dev_j_1, dev_j_2);

        // Realloc memory
        real2 *dev_a_new;
        real2 *dev_psi_1_new;
        real2 *dev_psi_2_new;
        real *dev_b_new;
        real2 *dev_j_1_new;
        real2 *dev_j_2_new;

        checkCudaErrors(cudaMalloc(&dev_a_new,     F_out*N*N*sizeof(real2)));
        checkCudaErrors(cudaMalloc(&dev_psi_1_new, F_out*N*N*sizeof(real2)));
        checkCudaErrors(cudaMalloc(&dev_psi_2_new, F_out*N*N*sizeof(real2)));
        checkCudaErrors(cudaMalloc(&dev_b_new,     F_out*N*N*sizeof(real)));
        checkCudaErrors(cudaMalloc(&dev_j_1_new,   F_out*N*N*sizeof(real2)));
        checkCudaErrors(cudaMalloc(&dev_j_2_new,   F_out*N*N*sizeof(real2)));

        redistributeLinear(handle, N, F, S, dev_a, dev_psi_1, dev_psi_2, F_out, dev_a_new, dev_psi_1_new, dev_psi_2_new);

        checkCudaErrors(cudaFree(dev_a));
        checkCudaErrors(cudaFree(dev_psi_1));
        checkCudaErrors(cudaFree(dev_psi_2));
        checkCudaErrors(cudaFree(dev_b));
        checkCudaErrors(cudaFree(dev_j_1));
        checkCudaErrors(cudaFree(dev_j_2));
        checkCudaErrors(cudaFree(s));
        checkCudaErrors(cudaFree(s_jb));
        checkCudaErrors(cudaFree(fenergy));

        dev_a = dev_a_new;
        dev_psi_1 = dev_psi_1_new;
        dev_psi_2 = dev_psi_2_new;
        dev_b = dev_b_new;
        dev_j_1 = dev_j_1_new;
        dev_j_2 = dev_j_2_new;

        checkCudaErrors(cudaMallocManaged(&s,   F_out*sizeof(real)));
        checkCudaErrors(cudaMallocManaged(&s_jb,   F_out*sizeof(real)));
        checkCudaErrors(cudaMallocManaged(&fenergy,   F_out*sizeof(real)));

        // Compute arc length
        // printf("System |  Delta s  | Delta sJB |\n");
        // for(int n=1; n<F_out; n++){
        //     printf("   %3d | %9.7f | %9.7f |\n", n, s[n] - s[n-1], s_jb[n] - s_jb\[n-1]);
        // }

        delete[] a;
        delete[] psi_1;
        delete[] psi_2;
        delete[] b;
        delete[] j_1;
        delete[] j_2;        

        a = new real2[F_out*N*N];
        psi_1 = new real2[F_out*N*N];
        psi_2 = new real2[F_out*N*N];
        b = new real[F_out*N*N];
        j_1 = new real2[F_out*N*N];
        j_2 = new real2[F_out*N*N];
    }
           
    computeFreeEnergy(F_out, dev_comp_nn_map, dev_sc_nn_map, 
        dev_a_1, dev_b_1, dev_m_xx_1, dev_m_yy_1, 
        dev_a_2, dev_b_2, dev_m_xx_2, dev_m_yy_2, 
        dev_h, dev_q_1, dev_q_2, dev_eta, dev_gamma, dev_delta,
        dev_a, dev_psi_1, dev_psi_2,
        fenergy);

    computeJB(F_out, dev_comp_nn_map, dev_sc_nn_map,
        dev_a_1, dev_b_1, dev_m_xx_1, dev_m_yy_1, 
        dev_a_2, dev_b_2, dev_m_xx_2, dev_m_yy_2, 
        dev_h, dev_q_1, dev_q_2,
        dev_a, dev_psi_1, dev_psi_2,
        dev_b, dev_j_1, dev_j_2);

    // Print final energy
    printf("┌──────┬────────────────┐ \n");
    printf("| Sys# |  Free Energy   | \n");        
    printf("├──────┼────────────────┤ \n");
    for(int n = 0; n<F; n+=max((F-1)/10,1)) 
        printf("|  %3.3d | %.8e | \n", n, fenergy[n]);
    printf("└──────┴────────────────┘ \n");

    // Copying back to host
    checkCudaErrors(cudaMemcpy(a, dev_a, F_out*N*N*sizeof(real2), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(psi_1, dev_psi_1, F_out*N*N*sizeof(real2), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(psi_2, dev_psi_2, F_out*N*N*sizeof(real2), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(b, dev_b, F_out*N*N*sizeof(real), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(j_1, dev_j_1, F_out*N*N*sizeof(real2), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(j_2, dev_j_2, F_out*N*N*sizeof(real2), cudaMemcpyDeviceToHost));

    // Save results
    printf("Saving results...  ");
    saveOutput(simulation_name, F_out, N, r, comp_domain, sc_domain, 
        a_1, b_1, m_xx_1, m_yy_1, a_2, b_2, m_xx_2, m_yy_2, h,
        q_1, q_2, eta, gamma, delta,
        psi_1, psi_2, a,
        j_1, j_2, b,
        fenergy, s, s_jb);
    printf("done.\n");

    // Free GPU memory
    checkCudaErrors(cudaFree(dev_comp_domain));
    checkCudaErrors(cudaFree(dev_comp_nn_map));

    checkCudaErrors(cudaFree(dev_sc_domain));
    checkCudaErrors(cudaFree(dev_sc_nn_map));

    checkCudaErrors(cudaFree(dev_a_1));
    checkCudaErrors(cudaFree(dev_b_1));
    checkCudaErrors(cudaFree(dev_m_xx_1));
    checkCudaErrors(cudaFree(dev_m_yy_1));

    checkCudaErrors(cudaFree(dev_a_2));
    checkCudaErrors(cudaFree(dev_b_2));
    checkCudaErrors(cudaFree(dev_m_xx_2));
    checkCudaErrors(cudaFree(dev_m_yy_2));

    checkCudaErrors(cudaFree(dev_q_1));
    checkCudaErrors(cudaFree(dev_q_2));
    checkCudaErrors(cudaFree(dev_eta));
    checkCudaErrors(cudaFree(dev_gamma));
    checkCudaErrors(cudaFree(dev_delta));

    checkCudaErrors(cudaFree(dev_h));
    
    checkCudaErrors(cudaFree(dev_psi_1));
    checkCudaErrors(cudaFree(dev_psi_2));
    checkCudaErrors(cudaFree(dev_a));

    checkCudaErrors(cudaFree(dev_j_1));
    checkCudaErrors(cudaFree(dev_j_2));
    checkCudaErrors(cudaFree(dev_b));

    checkCudaErrors(cudaFree(s));
    checkCudaErrors(cudaFree(s_jb));
    checkCudaErrors(cudaFree(fenergy));

    // Free host memory
    delete[] r;
    delete[] sc_domain;
    delete[] comp_domain;

    delete[] a_1;
    delete[] b_1;
    delete[] m_xx_1;
    delete[] m_yy_1;
        
    delete[] a_2;
    delete[] b_2;
    delete[] m_xx_2;
    delete[] m_yy_2;
        
    delete[] q_1;
    delete[] q_2;
    delete[] eta;    
    delete[] gamma;
    delete[] delta;

    delete[] h;

    delete[] psi_1;
    delete[] psi_2;
    delete[] a;   
    
    delete[] j_1;
    delete[] j_2;
    delete[] b;   

    printf("Finished\n");

    // Reset device and exit
    cudaDeviceReset();
    return 0;
}