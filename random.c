#include "random.h"
#include <stdlib.h>
#include <float.h>
#include <math.h>


// Using box-muller method to generate random normal numbers
// Credits: Wikipedia, ChatGPT and @mallocmyheart
float random_normal()
{
    static int have_spare = 0;
    static float z0, z1; 
    double u1, u2; // u1, u2 hold uniform random numbers
    
    if (have_spare)
    {
        have_spare = 0;
        return z1;
    }

    have_spare = 1;
    
    do
    {
        u1 = random() / (double) RAND_MAX;
    } while (u1 <= DBL_MIN);

    do
    {
        u2 = random() / (double) RAND_MAX;
    } while (u2 <= DBL_MIN);

    z0 = (float) (sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2));
    z1 = (float) (sqrt(-2.0 * log(u1)) * sin(2.0 * M_PI * u2));
    return z0;
}

