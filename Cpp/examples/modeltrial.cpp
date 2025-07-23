// following rs102 RooStats example
// RooFit/RooStats
#include "RooDataSet.h"
#include "RooRealVar.h"
#include "RooArgSet.h"
#include "RooGaussian.h"
#include "RooFFTConvPdf.h"
#include "RooAddPdf.h"
#include "RooPolynomial.h"
#include "RooAbsPdf.h"
#include "RooAbsReal.h"
#include "RooFitResult.h"
#include "RooMinimizer.h"
#include "RooPlot.h"
#include "RooAbsArg.h"
#include "RooAbsData.h"
#include "RooWorkspace.h"
#include "RooStats/ProfileLikelihoodCalculator.h"
#include "RooStats/HypoTestResult.h"

#include "TRint.h"
#include "TCanvas.h"
#include "TAxis.h"

// std
#include <iostream>
#include <iomanip>

// us
#include "RooBDNormalHPdf.h" 

// use this order for safety on library loading
using namespace RooFit;
using namespace RooStats;
 
// see below for implementation
void AddModel(RooWorkspace *);
void AddData(RooWorkspace *, int nev);
void DoHypothesisTest(RooWorkspace *);
void CheckPlots(RooWorkspace *);
 
//____________________________________
int main(int argc, char** argv)
{
  TRint app("app", &argc, argv);
  
  // The main macro.
  
  // Create a workspace to manage the project.
  RooWorkspace *wspace = new RooWorkspace("study");
  
  // one-off effort, book in wspace
  // add the signal and background models to the workspace
  AddModel(wspace);
  
  // use in loop for MC study unless is done automatically.
  // add some toy data to the workspace
  int nevents = 500; // vary for MC studies
  AddData(wspace, nevents);
  
  // fit
  RooAbsPdf *model = wspace->pdf("model");
  RooAbsData* data = wspace->data("toydata");
  RooRealVar *munu = wspace->var("munu");
  RooRealVar *en   = wspace->var("en");

  munu->setVal(0.0); // set lower than starter value helps fitting
  std::unique_ptr<RooFitResult> res{model->fitTo(*data, Save(true), PrintLevel(-1))};
  res->Print("v");

  // do the hypothesis test
  DoHypothesisTest(wspace);
  
  // one-off work at end of study.
  // make some plots
  CheckPlots(wspace);
  
  app.Run();
  // cleanup or save to disk
  delete wspace;
  // wspace->writeToFile("rs102.root")
  return 0;
}
 
//____________________________________
void AddModel(RooWorkspace *wspace)
{
 
    // Make models, Beta-Decay PDF combined with parameters
    // could have uncertainty PDF each, could be nuisance.
 
    double mnu    = 2.0e-4;
    double width  = 1.0e-3; // +-1 eV, under 1 MHz amplifier bandwidth
    double fixpt  = TBeta::endAt(mnu, 1);
    double ubound = fixpt + 0.5*width;
    double lbound = fixpt - 1.5*width; // [keV]
    double eresol = 5.0e-5;  // [keV]
    std::cout << "endpoint: " << std::setprecision(8) << fixpt << std::endl;

    // make PDF
    RooRealVar energy("en", "Energy", lbound, ubound, "keV");
    energy.setBins(4096, "cache");  // 'cache' for fft convolution

    RooRealVar munu("munu","neutrino mass", mnu, 0.0, 5.e-4, "keV"); // no negative mnu when fitting

    RooBDNormalHPdf spectrum("beta", "Beta Decay function", energy, munu);

    // --------------------------------------
    // make a Gauss resolution model.
    RooRealVar mu0("mu0", "zero", 0.0);
    RooRealVar eres("eres", "Energy resolution", eresol);
    RooGaussian resModel("resModel", "Resolution Model", energy, mu0, eres);
 
    // --------------------------------------
    // combined model
    // Construct convolution
    RooFFTConvPdf signal("signal","BD x Gauss",energy,spectrum,resModel) ;
    //    RooFFTConvPdf model("model","BD x Gauss",energy,spectrum,resModel) ;
    signal.setBufferFraction(0.2); // 20% of energy interval as buffer
    
    // Construct a flat pdf (polynomial of 0th order)
    RooPolynomial bckg("bckg", "flat background", energy);
 
    // Construct model = f*bckg + (1-f)*signal
    RooRealVar frac("frac", "fraction", 1.e-3, 0., 1.); // set tiny bckg contribution
    RooAddPdf model("model", "model", RooArgSet(bckg, signal), frac);
    
    wspace->import(model);
}
 
//____________________________________
void AddData(RooWorkspace *wks, int nev)
{
   // Add a toy dataset 
   RooAbsPdf *model = wks->pdf("model");
   RooRealVar *en = wks->var("en");
 
   std::unique_ptr<RooDataSet> data{model->generate(*en, nev)};
 
   wks->import(*data, Rename("toydata"));
}
 
//____________________________________
void DoHypothesisTest(RooWorkspace *wks)
{
 
   // Use a RooStats ProfileLikleihoodCalculator to do the hypothesis test.
   ModelConfig model;
   model.SetWorkspace(*wks);
   model.SetPdf("model");
 
   ProfileLikelihoodCalculator plc;
   plc.SetData(*(wks->data("toydata")));
 
   // here we explicitly set the value of the parameters for the null.
   // We want no signal contribution, eg. munu = 0
   RooRealVar *munu = wks->var("munu");
   
   RooArgSet poi(*munu);
   RooArgSet *nullParams = (RooArgSet *)poi.snapshot();
   nullParams->setRealValue("munu", 0.0);
 
   plc.SetModel(model);
   plc.SetNullParameters(*nullParams);
 
   // We get a HypoTestResult out of the calculator, and we can query it.
   HypoTestResult *htr = plc.GetHypoTest();
   std::cout << "-------------------------------------------------" << std::endl;
   std::cout << "The p-value for the null is " << htr->NullPValue() << std::endl;
   std::cout << "Corresponding to a significance of " << htr->Significance() << std::endl;
   std::cout << "-------------------------------------------------\n\n" << std::endl;
}
 
//____________________________________
void CheckPlots(RooWorkspace *wks)
{
  TCanvas* cv = new TCanvas("cv", "Beta Plot", 800, 600);
  cv->Divide(2);

  // plot PDF
  RooRealVar *en = wks->var("en");
  RooAbsPdf *model = wks->pdf("model");
  RooAbsPdf *bckg = wks->pdf("bckg");

  RooPlot* eframe = en->frame(Title("Beta Plot"));
  model->plotOn(eframe);
  
  // second plot
  RooPlot* eframe2 = en->frame(100); // restrict to 100 bins
  RooAbsData* data = wks->data("toydata");
  data->plotOn(eframe2);
  model->plotOn(eframe2);
  // Overlay the background component of model with a dashed line
  model->plotOn(eframe2, Components(*bckg), LineStyle(kDashed), LineColor(kRed));
 
  cv->cd(1);
  eframe->GetYaxis()->SetTitleOffset(1.5);
  eframe->Draw();
  cv->cd(2);
  eframe2->GetYaxis()->SetTitleOffset(1.5);
  eframe2->Draw();

  cv->Modified();
  cv->Update();
 
}
