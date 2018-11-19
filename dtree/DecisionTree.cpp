#include <iomanip>
#include <cmath>
#include <sstream>
#include "DecisionTree.hpp"

Feature::Feature(int size): size(size), data(new double[size]) {}

Feature::Feature(const Feature& copy): size(copy.size), data(new double[copy.size]) {
    int i(size);
    while(i--) {
        data[i] = copy.data[i];
    }
}

Feature::~Feature() {
    delete [] data;
}

int Feature::getSize() const {
    return size;
}

double Feature::operator[](const int& idx) const {
    return data[idx];
}

double& Feature::operator[](const int& idx) {
    return data[idx];
}

Feature& Feature::operator=(const Feature& f) {
    int i(size);
    while(i--) {
        data[i] = f.data[i];
    }
    return *this;
}

Feature Feature::operator-(const Feature& f) const {
    Feature res(*this);
    int i(size);
    while(i--) {
        res.data[i] -= f.data[i];
    }
    return res;
}

std::ostream& operator<<(std::ostream& out, const Feature& f) {
    out << "[";
    for (int i(0); i < f.size; ++i) {
        if (i) out << ", ";
        out << std::fixed << std::setw(9) << f.data[i];
    }
    out << "]";
    return out;
}

std::istream& operator>>(std::istream& in, Feature& f) {
    // Initialize
    int i(f.size);
    while (i--) f.data[i] = 0;
    std::stringstream ss, line;
    std::string input_line, buff;
    double value;
    int idx;
    getline(in, input_line);
    line << input_line;
    while (line.good()) {
        getline(line, buff, ':');
        ss << buff;
        ss >> idx;
        line >> value;
        f.data[idx - 1] = value;
        ss.str("");
        ss.clear();
    }
    return in;
}

Node::Node(int id, int num_attr, double threshold, Node* left_ptr, Node* right_ptr):
    id(id),
    num_attr(num_attr),
    threshold(threshold),
    left_ptr(left_ptr),
    right_ptr(right_ptr),
    l_id(-1),
    r_id(-1) {}

Node::~Node() {}

std::ostream& operator<< (std::ostream& out, const Node& node) {
    out << "+---------------------+" << std::endl;
    out << "|       Node " << std::setw(2) << node.id << "       |" << std::endl;
    out << "+---------------------+" << std::endl;
    out << "|   Attr " << node.num_attr << " >= " << std::left << std::setw(4) << std::setprecision(3) << node.threshold << " ?  |" << std::endl;
    out << "+---------------------+" << std::endl;
    out << "| Link  " << std::setw(2) << node.l_id << " | Link  " << std::setw(2) << node.r_id << " |" << std::endl;
    out << "| Class " << std::setw(1) << (node.l_cls == 1 ? "+1" : "-1") << " | Class " << std::setw(1) << (node.r_cls == 1 ? "+1" : "-1") << " |" << std::endl;
    out << "+---------------------+" << std::endl;
    return out;
}

void DecisionTree::_init_min_max_attr(const std::vector<Feature>& x, const std::vector<int>& t) {
    Feature buf;
    int j, i, size(x.size());
    int cnt[2] = {0};
    // init
    for (i = 0; i < 6; ++i) {
        attr_max_value[i] = -1e+100;
        attr_min_value[i] = 1e+100;
    }
    // Find max and min value
    for (j = 0; j < size; ++j) {
        buf = x[j];
        for (i = 0; i < 6; ++i) {
            attr_max_value[i] = attr_max_value[i] < buf[i] ? buf[i] : attr_max_value[i];
            attr_min_value[i] = attr_min_value[i] > buf[i] ? buf[i] : attr_min_value[i];
        }
        ++cnt[t[j]];
    }
    cls = cnt[0] < cnt[1];
    double gap;
    condition.clear();
    for (i = 0; i < 6; ++i) {
        condition.push_back(std::vector<double>());
        gap = attr_max_value[i] - attr_min_value[i] + 1;
        if (gap == 1) continue;
        switch ((int)gap) {
            case 2:
                condition[i].push_back(attr_max_value[i]);
                break;
            default:
                condition[i].push_back(attr_min_value[i] + gap / 3.);
                condition[i].push_back(attr_min_value[i] + gap * 2. / 3.);
                break;
        }
    }
}

double DecisionTree::_entropy(const int& num_attr, const double& threshold, const std::set<int>& idx,
                const std::vector<Feature>& x, const std::vector<int>& t,
                int* l_cnt, int* r_cnt) {
    l_cnt[0] = 0;
    l_cnt[1] = 0;
    r_cnt[0] = 0;
    r_cnt[1] = 0;
    std::set<int>::const_iterator it(idx.begin());
    for (; it != idx.end(); ++it) {
        if (x[*it][num_attr] < threshold)
            ++l_cnt[t[*it] == 1];
        else 
            ++r_cnt[t[*it] == 1];
    }
    if (!(l_cnt[0] + r_cnt[0]) || !(l_cnt[1] + r_cnt[1])) return 1.;
    double l_total (l_cnt[0]  + l_cnt[1]),
           r_total(r_cnt[0] + r_cnt[1]);
    double left_p0 (l_total ? l_cnt[0]  / l_total  : 0);
    double left_p1 (l_total ? l_cnt[1]  / l_total  : 0);
    double right_p0(r_total ? r_cnt[0] / r_total : 0);
    double right_p1(r_total ? r_cnt[1] / r_total : 0);
    return ( l_total * -(left_p0 * log(left_p0 + epsilon) 
           + left_p1 * log(left_p1 + epsilon)) 
           + r_total * -(right_p0 * log(right_p0 + epsilon) 
           + right_p1 * log(right_p1 + epsilon)))
           / (l_total + r_total) / log(2);
}

Node* DecisionTree::_build_tree(const std::vector<std::vector<double> >& c, const std::set<int>& idx,
                  const std::vector<Feature>& x, const std::vector<int>& t, int& id, int depth) {
    if (depth >= max_depth) { --id; return NULL; }
    int j(0), i;
    double min_entropy(1.), entropy;
    int num_attr, num_threshold;
    int l_tmp[2], r_tmp[2];
    int l_cnt[2], r_cnt[2];
    for (; j < 6; ++j) {
        for (i = 0; i < c[j].size(); ++i) {
            entropy = _entropy(j, c[j][i], idx, x, t, l_tmp, r_tmp);
            if (entropy < min_entropy) {
                min_entropy   = entropy;
                num_attr      = j;
                num_threshold = i;
                l_cnt[0]      = l_tmp[0];
                l_cnt[1]      = l_tmp[1];
                r_cnt[0]      = r_tmp[0];
                r_cnt[1]      = r_tmp[1];
            }
        }
    }
    if (min_entropy > entropy_threshold){ --id; return NULL; }
    double threshold(c[num_attr][num_threshold]);
    Node* tree_node = new Node(id, num_attr, threshold);
    std::set<int> l_idx;
    std::set<int> r_idx(idx);
    std::vector<std::vector<double> > l_c(c);
    std::vector<std::vector<double> > r_c(c);
    std::set<int>::iterator it(r_idx.begin());
    for (; it != r_idx.end(); ++it)
        if (x[*it][num_attr] < threshold) {
            l_idx.insert(*it);
            r_idx.erase(*it);
        }
    for (i = l_c[num_attr].size(); i > num_threshold; --i)
        l_c[num_attr].pop_back();
    for (i = 0; i <= num_threshold; ++i)
        r_c[num_attr].erase(r_c[num_attr].begin());
    tree_node->left_ptr  = _build_tree(l_c, l_idx, x, t, ++id, depth + 1);
    tree_node->right_ptr = _build_tree(r_c, r_idx, x, t, ++id, depth + 1);
    tree_node->l_cls     = l_cnt[0] < l_cnt[1] ? 1 : -1;
    tree_node->r_cls     = r_cnt[0] < r_cnt[1] ? 1 : -1;
    if (!tree_node->left_ptr && !tree_node->right_ptr && tree_node->l_cls == tree_node->r_cls) {
        --id;
        delete tree_node;
        return NULL;
    }
    if (tree_node->left_ptr)  tree_node->l_id = tree_node->left_ptr->id;
    if (tree_node->right_ptr) tree_node->r_id = tree_node->right_ptr->id;
    return tree_node;
}

void DecisionTree::_print_node(Node* ptr) {
    if (!ptr) return;
    std::cout << *ptr << std::endl;
    _print_node(ptr->left_ptr);
    _print_node(ptr->right_ptr);
}

DecisionTree::DecisionTree(double entropy_threshold, int max_depth, double epsilon):
    entropy_threshold(entropy_threshold),
    max_depth(max_depth),
    epsilon(epsilon) {}

DecisionTree::~DecisionTree() {}

void DecisionTree::fit(const std::vector<Feature>& x, const std::vector<int>& t) {
    int size(t.size());
    if (x.size() != size)
        throw "Input size no match";
    _init_min_max_attr(x, t);
    std::set<int> idx;
    for (int i(0); i < size; ++i)
        idx.insert(i);
    int id(0);
    root = _build_tree(condition, idx, x, t, id);
}

std::vector<int> DecisionTree::predict(const std::vector<Feature>& x) {
    int size(x.size());
    Node* ptr;
    int p;
    std::vector<int> res;
    for (int i(0); i < size; ++i) {
        ptr = root;
        p = cls;
        while (ptr) {
            if (x[i][ptr->num_attr] >= ptr->threshold) {
                p = ptr->r_cls;
                ptr = ptr->right_ptr;
            }
            else {
                p = ptr->l_cls;
                ptr = ptr->left_ptr;
            }
        }
        res.push_back(p);
    }
    return res;
}

void DecisionTree::print() {
    _print_node(root);
}

