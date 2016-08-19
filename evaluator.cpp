#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <set>
#include <iterator>
#include <cassert>
#include <thread>
#include "Eigen/Dense"
#include "utils.cpp"

#define uint unsigned int

using namespace Eigen;
using namespace std;

// returns soft (precision, recall, F1) per expression
// counts proportional overlap & binary overlap
Matrix<double, 3, 2>  testSequential(vector<vector<string> > &sents,
                                         vector<vector<string> > &labels) {
  uint nExprPredicted = 0;
  double nExprPredictedCorrectly = 0;
  uint nExprTrue = 0;
  double precNumerProp = 0, precNumerBin = 0;
  double recallNumerProp = 0, recallNumerBin = 0;
  for (uint i=0; i<sents.size(); i++) { // per sentence
    vector<string> labelsPredicted = sents[i];
    // To do: fill labelsPredicted
    string y, t, py="", pt="";
    uint match = 0;
    uint exprSize = 0;
    vector<pair<uint,uint> > pred, tru;
    int l1=-1, l2=-1;

    if (labels[i].size() != labelsPredicted.size())
      cout << labels[i].size() << " " << labelsPredicted.size() << endl;
    for (uint j=0; j<labels[i].size(); j++) { // per token in a sentence
      t = labels[i][j];
      y = labelsPredicted[j];

      if (t == "B") {
        //nExprTrue++;
        if (l1 != -1)
          tru.push_back(make_pair(l1,j));
        l1 = j;
      } else if (t == "I") {
        assert(l1 != -1);
      } else if (t == "O") {
        if (l1 != -1)
          tru.push_back(make_pair(l1,j));
        l1 = -1;
      } else
        assert(false);

      if ((y == "B") || ((y == "I") && ((py == "") || (py == "O")))) {
        nExprPredicted++;
        if (l2 != -1)
          pred.push_back(make_pair(l2,j));
        l2 = j;
      } else if (y == "I") {
        assert(l2 != -1);
      } else if (y == "O") {
        if (l2 != -1)
          pred.push_back(make_pair(l2,j));
        l2 = -1;
      } else {
        cout << y << endl;
        assert(false);
      }

      py = y;
      pt = t;
    }
    if ((l1 != -1) && (l1 != labels[i].size()))
      tru.push_back(make_pair(l1,labels[i].size()));
    if ((l2 != -1) && (l2 != labels[i].size()))
      pred.push_back(make_pair(l2,labels[i].size()));

    vector<bool> trum = vector<bool>(tru.size(),false);
      vector<bool> predm = vector<bool>(pred.size(),false);
    for (uint a=0; a<tru.size(); a++) {
      pair<uint,uint> truSpan = tru[a];
      nExprTrue++;
      for (uint b=0; b<pred.size(); b++) {
        pair<uint,uint> predSpan = pred[b];

        uint lmax, rmin;
        if (truSpan.first > predSpan.first)
          lmax = truSpan.first;
        else
          lmax = predSpan.first;
        if (truSpan.second < predSpan.second)
          rmin = truSpan.second;
        else
          rmin = predSpan.second;

        uint overlap = 0;
        if (rmin > lmax)
          overlap = rmin-lmax;

        if (predSpan.second == predSpan.first) cout << predSpan.first << endl;
        assert(predSpan.second != predSpan.first);
        precNumerProp += (double)overlap/(predSpan.second-predSpan.first);
        recallNumerProp += (double)overlap/(truSpan.second-truSpan.first);

        bool match = false;
        if ((truSpan.first == predSpan.first) && (truSpan.second == predSpan.second))
            match = true;

        if (!predm[b] && match) {
          precNumerBin += 1;
          predm[b] = true;
        }
        if (!trum[a] && match) {
          recallNumerBin += 1;
          trum[a]=true;
        }
      }
    }

  }
  double precisionProp = (nExprPredicted==0) ? 1 : precNumerProp/nExprPredicted;
  double recallProp = recallNumerProp/nExprTrue;
  double f1Prop = (2*precisionProp*recallProp)/(precisionProp+recallProp);
  double precisionBin = (nExprPredicted==0) ? 1 : precNumerBin/nExprPredicted;
  double recallBin = recallNumerBin/nExprTrue;
  double f1Bin = (2*precisionBin*recallBin)/(precisionBin+recallBin);
  Matrix<double, 3, 2> results;
  results << precisionProp, precisionBin,
             recallProp, recallBin,
             f1Prop, f1Bin;
  return results;
}

// Read labels from file
void readLabels(vector<vector<string> > &T, string fname) {
  ifstream in(fname.c_str());
  string line;
  vector<string> t; // individual sentences and labels
  while(std::getline(in, line)) {
    if (isWhitespace(line)) {
      if (t.size() != 0) {
        T.push_back(t);
        t.clear();
      }
    } else {
      string token, part, label;
      uint i = line.find_first_of(" \t");
      token = line.substr(0, i);
      uint j = line.find_first_of(" \t", i+1);
      part = line.substr(i+1,j-i-1);
      //cout << part << endl;
      i = line.find_last_of(" \t");
      label = line.substr(i+1, 1);
      t.push_back(label);
    }
  }
  if (t.size() != 0) {
    T.push_back(t);
    t.clear();
  }
}

// Main Function
int main(int argc, char **argv) {
  // Parse argv
  string predictFile = argv[1];
  string truthFile = argv[2];

  vector<vector<string> > predicted, truth;
  readLabels(predicted, predictFile);
  readLabels(truth, truthFile);

  Matrix<double, 3, 2> result = testSequential(predicted, truth);
  cout << result << endl;

  return 0;
}

