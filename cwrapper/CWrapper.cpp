#include <vector>
#include "DecisionTree.hpp"

extern "C" {
    Feature* newFeature(int len) {
        return new Feature(len);
    }
    void FeatureSet(Feature arr, int idx, double value) {
        arr[idx] = value;
    }
    int FeatureLen(Feature arr) {
        return arr.getSize();
    }
    double FeatureRead(Feature arr, int idx) {
        return arr[idx];
    }
    std::vector<int>* newVectorInt() {
        return new std::vector<int>();
    }
    void VectorIntPush(std::vector<int>* vec, int value) {
        vec->push_back(value);
    }
    int VectorIntRead(std::vector<int>* vec, int idx) {
        return (*vec)[idx];
    }
    int VectorIntLen(std::vector<int>* vec) {
        return vec->size();
    }
    std::vector<Feature>* newVectorFeature() {
        return new std::vector<Feature>();
    }
    void VectorFeaturePush(std::vector<Feature>* vec, Feature value) {
        vec->push_back(value);
    }
    int VectorFeatureLen(std::vector<Feature>* vec) {
        return vec->size();
    }
    Feature* VectorFeatureRead(std::vector<Feature> vec, int idx) {
        return new Feature(vec[idx]);
    }
    DecisionTree* newDecisionTree(double threshold, int max_depth) {
        return new DecisionTree(threshold, max_depth);
    }
    void fit(DecisionTree* dt, std::vector<Feature>* x, std::vector<int>* t) {
        dt->fit(*x, *t);
    }
    std::vector<int>* predict(DecisionTree* dt, std::vector<Feature>* x) {
        return new std::vector<int>(dt->predict(*x));
    }
    void printTree(DecisionTree* dt) {
        dt->print();
    }
}
