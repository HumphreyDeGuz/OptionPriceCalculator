#include <BinomialTree.h>
#include <TreeAmerican.h>
#include <TreeEuropean.h>
#include <BlackScholesFormulas.h>
#include <PayOffForward.h>
#include <omp.h>
#include <iostream>
#include <ctime>
#include <cmath>

using namespace std;

int main()
{
    double Expiry = 365*2;
    double Strike = 125;
    double Spot = 120;
    double Vol = 0.10;
    double r = 0.05;
    double d = 0.0;
    unsigned long Steps = 24 * (unsigned long)Expiry; // Every hour until expiration

    int start_s = clock();

    PayOffCall thePayOff(Strike);

    ParametersConstant rParam(r);
    ParametersConstant dParam(d);

    TreeEuropean europeanOption(Expiry,thePayOff);
    TreeAmerican americanOption(Expiry,thePayOff);

    SimpleBinomialTree theTree(Spot,rParam,dParam,Vol,Steps,Expiry);
    double euroPrice = theTree.GetThePrice(europeanOption);
    double americanPrice = theTree.GetThePrice(americanOption);
    cout << "\nSimulation Results:\n";
    cout << "European price: " << euroPrice;
    cout <<"\nAmerican price: " << americanPrice << "\n";

    double BSPrice = BlackScholesCall(Spot,Strike,r,d,Vol,Expiry);
    cout << "\nBS formula results: \n";
    cout << "European price: " << BSPrice << "\n";

    PayOffForward forwardPayOff(Strike);
    TreeEuropean forward(Expiry,forwardPayOff);

    double forwardPrice = theTree.GetThePrice(forward);
    cout << "\nForward price by tree: " << forwardPrice;

    double actualForwardPrice = exp(-r*Expiry)*(Spot*exp((r-d)*Expiry)-Strike);
    cout << "\nForward price: " << actualForwardPrice << "\n";

    int stop_s = clock();
    cout << "\nTime: " << (stop_s - start_s) / double(CLOCKS_PER_SEC) << " seconds\n";

#pragma omp parallel
    {
        start_s = clock();
        euroPrice = theTree.GetThePrice(europeanOption);
        americanPrice = theTree.GetThePrice(americanOption);
        cout << "\nSimulation Results w/OpenMP:\n";
        cout << "European price: " << euroPrice;
        cout <<"\nAmerican price: " << americanPrice << "\n";

        BSPrice = BlackScholesCall(Spot,Strike,r,d,Vol,Expiry);
        cout << "\nBS formula results: \n";
        cout << "European price: " << BSPrice << "\n";

        forwardPrice = theTree.GetThePrice(forward);
        cout << "\nForward price by tree: " << forwardPrice;

        actualForwardPrice = exp(-r*Expiry)*(Spot*exp((r-d)*Expiry)-Strike);
        cout << "\nForward price: " << actualForwardPrice << "\n";

        stop_s = clock();
        cout << "\nTime: " << (stop_s - start_s) / double(CLOCKS_PER_SEC) << " seconds\n";
    }

    double Open;
    double Closed;
    double difference = 0;

#pragma omp parallel for
    {
        for(int i = 0; i < 5; ++i){
            start_s = clock();
            euroPrice = theTree.GetThePrice(europeanOption);
            americanPrice = theTree.GetThePrice(americanOption);
            BSPrice = BlackScholesCall(Spot,Strike,r,d,Vol,Expiry);
            stop_s = clock();
            Open = (stop_s -start_s)/double(CLOCKS_PER_SEC);
#pragma omp critical
            {
                start_s = clock();
                euroPrice = theTree.GetThePrice(europeanOption);
                americanPrice = theTree.GetThePrice(americanOption);
                BSPrice = BlackScholesCall(Spot,Strike,r,d,Vol,Expiry);
                stop_s = clock();
                Closed = (stop_s -start_s)/double(CLOCKS_PER_SEC);
            }
            difference += (Open-Closed);
        }
    }
    difference /= 10;

    cout << "\nThe average of the difference between OpenMP and Regular sims are " << difference << " seconds.\n";



    return 0;
}