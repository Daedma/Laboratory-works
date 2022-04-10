#include <fstream>
#include <queue>
#include <iterator>

class tree
{
    struct node
    {
        int val;
        node* left;
        node* right;
        node(int v = 0, node* l = nullptr, node* r = nullptr) noexcept : val { v }, left { l }, right { r } {}
        void clear_child() noexcept
        {
            if (left)
            {
                left->clear_child();
                delete left;
            }
            if (right)
            {
                right->clear_child();
                delete right;
            }
        }
        void reflect() noexcept
        {
            using std::swap;
            if (left) left->reflect();
            if (right) right->reflect();
            swap(left, right);
        }
        void print(std::ostream& os) const
        {
            os << val << " ";
            if (left) left->print(os);
            if (right) right->print(os);
        }
    };
    node* root = nullptr;
public:
    tree() = default;

    template<typename InputIt>
    tree(InputIt first, InputIt last)
    {
        if (first == last) return;
        root = new node(*first++, nullptr, nullptr);
        std::queue<node*> q;
        q.emplace(root);
        while (first != last)
        {
            q.front()->left = new node(*first++, nullptr, nullptr);
            if (first == last) return;
            q.front()->right = new node(*first++, nullptr, nullptr);
            q.emplace(q.front()->left);
            q.emplace(q.front()->right);
            q.pop();
        }
    }

    ~tree() noexcept
    {
        if (root)
        {
            root->clear_child();
            delete root;
        }
    }

    void reflect() noexcept
    {
        if (root) root->reflect();
    }

    std::ostream& print(std::ostream& os) const
    {
        if (root) root->print(os);
        return os;
    }
};

std::ostream& operator<<(std::ostream& os, const tree& rhs)
{
    return rhs.print(os);
}

int main()
{
    std::ifstream ifs { "input.txt" };
    tree balanced_tree { std::istream_iterator<int>{ifs}, {} };
    ifs.close();
    balanced_tree.reflect();
    std::ofstream ofs { "output.txt" };
    ofs << balanced_tree;
    ofs.close();
}