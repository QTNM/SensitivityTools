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
#include "RooAbsArg.h"
#include "RooAbsData.h"
#include "RooWorkspace.h"
#include "RooMCStudy.h"
#include "RooRandomizeParamMCSModule.h"

//#include "TApplication.h"
#include "TCanvas.h"
#include "TAxis.h"
#include "TH1.h"
#include "TTree.h"
#include "TFile.h"

// std
#include <iostream>
#include <iomanip>

// us
#include "RooBDNormalHPdf.h" 
#include "CLI11.hpp"

using namespace RooFit;
 
struct Config {
  double mnull = 0.0;
  double munu = 2.0e-4; // [keV]
  double awidth = 1.0e-3; // analysis width [keV]
  double resolution = 5.0e-5; // [keV]
};

// see below for implementation
void AddModel(RooWorkspace *, Config&);
void DoMC(RooWorkspace *, TFile*, int, int);
 
//____________________________________
int main(int argc, char** argv)
{
  // command line interface
  CLI::App app{"MC app"};
  Config cfg;
  
  int nevents = -1; // must be set on CL
  int nsims = -1; // must be set on CL
  double mn = 2.0e-4;
  double eres = 5.0e-5;
  double width = 1.0e-3;
  std::string outfname = "mcout.root";

  app.add_option("-n,--nevents", nevents, "<mean number of events> Default: -1");
  app.add_option("-s,--nsims", nsims, "<number of simulations> Default: -1");
  app.add_option("-m,--mass", mn, "<lightest neutrino mass (keV)> Default: 2.0e-4");
  app.add_option("-r,--eres", eres,  "<energy resolution sigma (keV)> Default: 5.0e-5");
  app.add_option("-w,--awidth", width,  "<analysis window energy width (keV)> Default: 1.0e-3");
  app.add_option("-o,--output", outfname, "<output file name> Default: mcout.root");

  CLI11_PARSE(app, argc, argv);

  if (nevents<0) {
    std::cout << "Error: Needs mean number of events to simulate on command line, option -n" << std::endl;
    return 0;
  }
  if (nsims<0) {
    std::cout << "Error: Needs number of simulations on command line, option -s" << std::endl;
    return 0;
  }
  if (mn>0.0) cfg.munu = mn;
  else std::cout << "Error: neutrino mass > 0 required, uses default" << std::endl;
  if (eres>0.0) cfg.awidth = width;
  else std::cout << "Error: energy analysis window width > 0 required, uses default" << std::endl;
  if (eres>0.0) cfg.resolution = eres;
  else std::cout << "Error: energy resolution > 0 required, uses default" << std::endl;
  
  // The main macro.
  TFile* fout = new TFile(outfname.data(), "RECREATE");
  
  // Create a workspace to manage the project.
  RooWorkspace *wspace = new RooWorkspace("study");
  
  // one-off effort, book in wspace
  // add the signal and background models to the workspace
  AddModel(wspace, cfg);
  
  // do the Monte-Carlo
  DoMC(wspace, fout, nevents, nsims); // write TTrees in file
  fout->Close();
  
  //  app.Run();


  // cleanup or save to disk
  delete wspace;
  // wspace->writeToFile("rs102.root")
  return 0;
}
 
//____________________________________
void AddModel(RooWorkspace *wspace, Config& cf)
{
 
    // Make models, Beta-Decay PDF combined with parameters
    // could have uncertainty PDF each, could be nuisance.
 
  double mnu    = cf.munu;
  double mnull  = cf.mnull;
  double width  = cf.awidth; // analysis width; +-1 eV, under 1 MHz amplifier bandwidth
  double eresol = cf.resolution;  // Gauss sigma [keV]

  double fixpt  = TBeta::endAt(mnu, 1);
  double fixptn = TBeta::endAt(mnull, 1);
  double ubound = fixpt + 0.5*width;
  double lbound = fixpt - 1.5*width; // [keV]

  std::cout << "endpoint: " << std::setprecision(8) << fixpt << std::endl;
  std::cout << "endpoint null: " << std::setprecision(8) << fixptn << std::endl;
  
  // make PDF
  RooRealVar energy("en", "Energy", lbound, ubound, "keV");
  energy.setBins(4096, "cache");  // 'cache' for fft convolution
  
  RooRealVar munu("munu","neutrino mass", mnu, 0.0, 5.e-4, "keV"); // no negative mnu when fitting
  RooRealVar munull("munull","neutrino mass", mnull, 0.0, 5.e-4, "keV"); // no negative mnu when fitting
  
  RooBDNormalHPdf spectrum("beta", "Beta Decay function", energy, munu);
  RooBDNormalHPdf null("mnull", "Beta Decay function", energy, munull);
  
  // --------------------------------------
  // make a Gauss resolution model.
  RooRealVar mu0("mu0", "zero", 0.0);
  RooRealVar eres("eres", "Energy resolution", eresol); // const
  RooGaussian resModel("resModel", "Resolution Model", energy, mu0, eres);
  RooGaussian resModel2("resModel2", "Resolution Model", energy, mu0, eres);
  
  // --------------------------------------
  // combined model
  // Construct convolution
  RooFFTConvPdf signal("signal","BD x Gauss",energy,spectrum,resModel) ;
  signal.setBufferFraction(0.2); // 20% of energy interval as buffer
  
  RooFFTConvPdf nosignal("nosignal","BD x Gauss",energy,null,resModel2) ;
  nosignal.setBufferFraction(0.2); // 20% of energy interval as buffer
  
  // Construct a flat pdf (polynomial of 0th order)
  RooPolynomial bckg("bckg", "flat background", energy);
  RooPolynomial bckg2("bckg2", "flat background", energy);
  
  // Construct model = f*bckg + (1-f)*signal
  RooRealVar frac("frac", "fraction", 1.e-3, 0., 1.); // set tiny bckg contribution
  RooAddPdf model("model", "model", RooArgList(bckg, signal), frac);
  
  RooRealVar frac2("frac2", "fraction", 1.e-3, 0., 1.); // set tiny bckg contribution
  RooAddPdf h0("h0", "H0", RooArgList(bckg2, nosignal), frac2);
  
  wspace->import(model);
  wspace->import(h0);
}
 
 
//____________________________________
void DoMC(RooWorkspace *wks, TFile* ff, int nevents, int nsims)
{
  RooAbsPdf *model = wks->pdf("model");
  RooAbsPdf *h0    = wks->pdf("h0");
  RooRealVar *en   = wks->var("en");

  // Configure manager to perform binned extended likelihood fits (Binned(),Extended()) on data generated
  // with a Poisson fluctuation on Nobs (Extended())
  RooMCStudy *mcs = new RooMCStudy(*model, *en, Extended(true), Silence(), 
				   FitOptions(Save(true), PrintEvalErrors(-1)));
  RooMCStudy *mcs2= new RooMCStudy(*h0, *en, Extended(true), Silence(), 
				   FitOptions(Save(true), PrintEvalErrors(-1)));
 
  // R u n   m a n a g e r 
  // ----------------------
  
  // Run N experiments, M data samples drawn.
  mcs->generateAndFit(nsims, nevents);
  //  std::cout << "H0 sim:" << std::endl;
  mcs2->generateAndFit(nsims, nevents);

  // all variables in ttree
  TTree* mcstree = mcs->fitParDataSet().GetClonedTree();
  TTree* mcs2tree = mcs2->fitParDataSet().GetClonedTree();
  mcstree->SetName("wmass");
  mcs2tree->SetName("hnull");
  mcstree->SetDirectory(ff);
  mcs2tree->SetDirectory(ff);
  mcstree->Write();
  mcs2tree->Write();
}
