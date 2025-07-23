//
// Use the qtnmModel for sensitivity analysis
//

// std libs
#include <iostream>
#include <iomanip>

// us
#include "RooKurieNHPdf.h" // for compilation

// ROOT
#include "TRint.h"
#include "TCanvas.h"
#include "TAxis.h"
#include "RooRealVar.h"
#include "RooDataSet.h"
#include "RooPlot.h"

// ROOFIT
using namespace RooFit;

int main(int argc, char** argv)
{
  TRint app("app", &argc, argv);
  TCanvas* cv = new TCanvas("cv", "Kurie Plot", 800, 600);
  cv->Divide(2);
  
  double mnu    = 0.0; // null hypothesis
  double ubound = TBeta::endAt(mnu, 1);
  double lbound = ubound - 0.001; // [keV]
  std::cout << "upper limit: " << std::setprecision(8) << ubound << std::endl;

  // make PDF
  RooRealVar en("en", "Energy [keV]", lbound, ubound);
  RooRealVar munu("munu","neutrino mass", mnu, 0.0, 1.e-3); // do not allow for negative mnu when fitting
  RooKurieNHPdf spec("kurie", "Kurie function", en, munu);
  
  // plot PDF
  RooPlot* eframe = en.frame(Title("Kurie Plot"));
  spec.plotOn(eframe);
  
  // second plot
  munu.setVal(2.0e-4);
  std::unique_ptr<RooDataSet> data{spec.generate(en, 1000)};
  RooPlot* eframe2 = en.frame(Title("Kurie Plot with data"));
  data->plotOn(eframe2);
  spec.plotOn(eframe2);
  
  cv->cd(1);
  eframe->GetYaxis()->SetTitleOffset(1.4);
  eframe->Draw();
  cv->cd(2);
  eframe2->GetYaxis()->SetTitleOffset(1.4);
  eframe2->Draw();

  cv->Modified();
  cv->Update();
  app.Run();
  return 0;
}
