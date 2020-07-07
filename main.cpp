#include <iostream>
#include <math.h>


double sigmoid(double ln) {
    auto out = 1.0/(1.0 + exp(-ln));      
    return out;
}

double sigmoid_derivative(double ln) {
    return sigmoid(ln)*(1-sigmoid(ln));
}

double rando() {
   return  ((double)rand()/((double)RAND_MAX+1));
}

#define NUMPAT 4
#define NUMIN  2
#define NUMHID 2
#define NUMOUT 1

int main(int, char**) {
    std::cout << "One Hidden lay neural network!\n";

    int ranpat[NUMPAT+1];
    int NumPattern = NUMPAT, NumInput = NUMIN, NumHidden = NUMHID, NumOutput = NUMOUT;
    double Input[NUMPAT+1][NUMIN+1] = { {0, 0, 0},  {0, 0, 0},  {0, 1, 0},  {0, 0, 1},  {0, 1, 1} };
    double Target[NUMPAT+1][NUMOUT+1] = { {0, 0},  {0, 0},  {0, 1},  {0, 1},  {0, 0} };
    double SumH[NUMPAT+1][NUMHID+1], WeightIH[NUMIN+1][NUMHID+1], Hidden[NUMPAT+1][NUMHID+1];
    double SumO[NUMPAT+1][NUMOUT+1], WeightHO[NUMHID+1][NUMOUT+1], Output[NUMPAT+1][NUMOUT+1];
    double DeltaO[NUMOUT+1], SumDOW[NUMHID+1], DeltaH[NUMHID+1];
    double DeltaWeightIH[NUMIN+1][NUMHID+1], DeltaWeightHO[NUMHID+1][NUMOUT+1];
    double error, eta = 0.5, alpha = 0.9, smallwt = 0.5;
  
    for(int j = 1 ; j <= NumHidden ; j++ ) {    /* initialize WeightIH and DeltaWeightIH */
        for(int  i = 0 ; i <= NumInput ; i++ ) { 
            DeltaWeightIH[i][j] = 0.0 ;
            WeightIH[i][j] = 2.0 * ( rando() - 0.5 ) * smallwt ;
        }
    }
    for(int  k = 1 ; k <= NumOutput ; k ++ ) {    /* initialize WeightHO and DeltaWeightHO */
        for(int j = 0 ; j <= NumHidden ; j++ ) {
            DeltaWeightHO[j][k] = 0.0 ;              
            WeightHO[j][k] = 2.0 * ( rando() - 0.5 ) * smallwt ;
        }
    }
    int epoch = 0;
    while (epoch < 100000) {    /* iterate weight updates */
        for(int p = 1 ; p <= NumPattern ; p++ ) {    /* randomize order of training patterns */
            ranpat[p] = p ;
        }
        for(int p = 1 ; p <= NumPattern ; p++) {
            int np = p + rando() * ( NumPattern + 1 - p ) ;
            int op = ranpat[p] ; ranpat[p] = ranpat[np] ; ranpat[np] = op ;
        }
        error = 0.0 ;

        for(int np = 1 ; np <= NumPattern ; np++ ) {    /* repeat for all the training patterns */
            int p = ranpat[np];
            for(int j = 1 ; j <= NumHidden ; j++ ) {    /* compute hidden unit activations */
                SumH[p][j] = WeightIH[0][j] ;
                for(int i = 1 ; i <= NumInput ; i++ ) {
                    SumH[p][j] += Input[p][i] * WeightIH[i][j] ;
                }
                Hidden[p][j] = 1.0/(1.0 + exp(-SumH[p][j])) ;
            }
            for(int k = 1 ; k <= NumOutput ; k++ ) {    /* compute output unit activations and errors */
                SumO[p][k] = WeightHO[0][k] ;
                for(int j = 1 ; j <= NumHidden ; j++ ) {
                    SumO[p][k] += Hidden[p][j] * WeightHO[j][k] ;
                }
                Output[p][k] = 1.0/(1.0 + exp(-SumO[p][k])) ;   /* Sigmoidal Outputs */
/*              Output[p][k] = SumO[p][k];      Linear Outputs */
                error += 0.5 * (Target[p][k] - Output[p][k]) * (Target[p][k] - Output[p][k]) ;   /* SSE */
/*              error -= ( Target[p][k] * log( Output[p][k] ) + ( 1.0 - Target[p][k] ) * log( 1.0 - Output[p][k] ) ) ;    Cross-Entropy error */
                DeltaO[k] = (Target[p][k] - Output[p][k]) * Output[p][k] * (1.0 - Output[p][k]) ;   /* Sigmoidal Outputs, SSE */
/*              DeltaO[k] = Target[p][k] - Output[p][k];     Sigmoidal Outputs, Cross-Entropy error */
/*              DeltaO[k] = Target[p][k] - Output[p][k];     Linear Outputs, SSE */
            }
            for(int  j = 1 ; j <= NumHidden ; j++ ) {    /* 'back-propagate' errors to hidden layer */
                SumDOW[j] = 0.0 ;
                for(int  k = 1 ; k <= NumOutput ; k++ ) {
                    SumDOW[j] += WeightHO[j][k] * DeltaO[k] ;
                }
                DeltaH[j] = SumDOW[j] * Hidden[p][j] * (1.0 - Hidden[p][j]) ;
            }
            for(int j = 1 ; j <= NumHidden ; j++ ) {     /* update weights WeightIH */
                DeltaWeightIH[0][j] = eta * DeltaH[j] + alpha * DeltaWeightIH[0][j] ;
                WeightIH[0][j] += DeltaWeightIH[0][j] ;
                for(int i = 1 ; i <= NumInput ; i++ ) { 
                    DeltaWeightIH[i][j] = eta * Input[p][i] * DeltaH[j] + alpha * DeltaWeightIH[i][j];
                    WeightIH[i][j] += DeltaWeightIH[i][j] ;
                }
            }
            for(int k = 1 ; k <= NumOutput ; k ++ ) {    /* update weights WeightHO */
                DeltaWeightHO[0][k] = eta * DeltaO[k] + alpha * DeltaWeightHO[0][k] ;
                WeightHO[0][k] += DeltaWeightHO[0][k] ;
                for(int j = 1 ; j <= NumHidden ; j++ ) {
                    DeltaWeightHO[j][k] = eta * Hidden[p][j] * DeltaO[k] + alpha * DeltaWeightHO[j][k] ;
                    WeightHO[j][k] += DeltaWeightHO[j][k] ;
                }
            }
        }
        if( epoch%100 == 0 )  {
            std::fprintf(stdout, "\nEpoch %-5d :   error = %f", epoch, error) ;
        }
        if( error < 0.0004 ) {
            break ;  /* stop learning when 'near enough' */
        } 
        epoch++;
    }
    
    std::fprintf(stdout, "\n\nStopped after Epoch %d\n\nPat\t", epoch) ;   /* print network outputs */

    for(int i = 1 ; i <= NumInput ; i++ ) {
        std::fprintf(stdout, "Input%-4d\t", i) ;
    }
    for(int k = 1 ; k <= NumOutput ; k++ ) {
        std::fprintf(stdout, "Target%-4d\tOutput%-4d\t", k, k) ;
    }
    for(int p = 1 ; p <= NumPattern ; p++ ) {        
        std::fprintf(stdout, "\n%d\t", p) ;
        for(int i = 1 ; i <= NumInput ; i++ ) {
            std::fprintf(stdout, "%f\t", Input[p][i]) ;
        }
        for(int k = 1 ; k <= NumOutput ; k++ ) {
            std::fprintf(stdout, "%f\t%f\t", Target[p][k], Output[p][k]) ;
        }
    }
    std::cout << "\n\nGoodbye!\n\n";
    return 1 ;
}
 