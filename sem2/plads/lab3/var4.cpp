#include <fstream>
#include <queue>
#include <iterator>

class tree//Класс дерева
{
    struct node//Структура узла
    {
        int val;//Значение в узле
        node* left;//Левый потомок
        node* right;//Правый потомок
        node(int v = 0, node* l = nullptr, node* r = nullptr) noexcept : val { v }, left { l }, right { r } {}
        void clear_child() noexcept//Удалить всех потомков данного узла
        {
            if (left)
            {
                left->clear_child();
                delete left;
                left = nullptr;
            }
            if (right)
            {
                right->clear_child();
                delete right;
                right = nullptr;
            }
        }
        void reflect() noexcept//Отразить ветку относительно текущего узла
        {
            using std::swap;
            if (left) left->reflect();
            if (right) right->reflect();
            swap(left, right);
        }
        void print(std::ostream& os) const//Вывести дерево прямым обходом
        {
            os << val << " ";
            if (left) left->print(os);
            if (right) right->print(os);
        }
    };

    node* root = nullptr;//Корень дерева
public:
    tree() = default;

    template<typename InputIt>
    tree(InputIt first, InputIt last)//Конструктор для инициализации дерева диапозоном значений
    {
        if (first == last) return;
        root = new node(*first++);
        std::queue<node*> q;
        q.emplace(root);
        while (first != last)
        {
            q.front()->left = new node(*first++);
            if (first == last) return;
            q.front()->right = new node(*first++);
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

    void reflect() noexcept//Метод отражения дерева относительно корня
    {
        if (root) root->reflect();
    }

    std::ostream& print(std::ostream& os) const//Вывод дерева с помощью прямого обхода
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