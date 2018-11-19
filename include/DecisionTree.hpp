#ifndef __DecisionTree__
#define __DecisionTree__
#include <iostream>
#include <vector>
#include <set>

class Feature {
    double* data;
    int     size;
public:
    Feature(int size=6);
    Feature(const Feature& copy);
    ~Feature();
    int getSize() const;
    double operator[](const int& idx) const;
    double& operator[](const int& idx);
    Feature& operator=(const Feature& f);
    Feature operator-(const Feature& f) const;
    friend std::ostream& operator<<(std::ostream& out, const Feature& f);
    friend std::istream& operator>>(std::istream& in, Feature& f);
};

class Node {
    friend class DecisionTree;
    int      id;
    int      num_attr;
    double   threshold;
    int      l_cls;
    int      r_cls;
    int      l_id;
    int      r_id;
    Node*    left_ptr;
    Node*    right_ptr;
public:
    Node(int id=-1, int num_attr=0, double threshold=0, Node* left_ptr=NULL, Node* right_ptr=NULL);
    ~Node();
    friend std::ostream& operator<< (std::ostream& out, const Node& node);
};

class DecisionTree {
    double                  entropy_threshold;
    int                     max_depth;
    Feature                 attr_max_value;
    Feature                 attr_min_value;
    std::vector<std::vector<double> > condition;
    Node*                   root;
    double                  epsilon;
    int                     cls;
    void _init_min_max_attr(const std::vector<Feature>& x, const std::vector<int>& t);
    double _entropy(const int& num_attr, const double& threshold, const std::set<int>& idx,
                    const std::vector<Feature>& x, const std::vector<int>& t,
                    int* l_cnt, int* r_cnt);
    Node* _build_tree(const std::vector<std::vector<double> >& c, const std::set<int>& idx,
                      const std::vector<Feature>& x, const std::vector<int>& t,
                      int& id, int depth=0);
    void _print_node(Node* ptr);
public:
    DecisionTree(double entropy_threshold=.5, int max_depth=5, double epsilon=.00001);
    ~DecisionTree();
    void fit(const std::vector<Feature>& x, const std::vector<int>& t);
    std::vector<int> predict(const std::vector<Feature>& x);
    void print();
};
#endif
