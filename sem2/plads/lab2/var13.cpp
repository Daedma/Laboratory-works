#include <fstream>
#include <iostream>
#include <utility>

struct list
{
    double val;
    list* next;
};

//вставка элемента после _lst со значением val
list* insert(list* _lst, double val)
{
    return _lst->next = new list { val, _lst->next };
}

//удаление элемента после _lst
list* erase(list* _lst)
{
    //(*_lst).next = _lst->next
    list* next = _lst->next->next;//сохраняем указатель на следующий за удаляемым элементом
    delete _lst->next;//удаляем элемент
    return _lst->next = next;//текущий элемент теперь должен ссылаться на следующий за удаленным
}

size_t count(const char* aFile)//Number of items in the file
{
    std::ifstream ifs { aFile };
    size_t num = 0;
    double d;
    while (ifs >> d) ++num;
    ifs.close();
    return num;
}

void swap(list* lhs, list* rhs)
{
    if (rhs->next == lhs)
    {
        rhs->next = lhs->next;
        lhs->next = rhs->next->next;
        rhs->next->next = lhs;
        return;
    }
    else if (lhs->next == rhs)
    {
        swap(rhs, lhs);
        return;
    }
    list* old_lcur = lhs->next;
    list* old_rcur = rhs->next;
    list* old_lnext = lhs->next->next;
    list* old_rnext = rhs->next->next;
    lhs->next->next = old_rnext;
    rhs->next->next = old_lnext;
    lhs->next = old_rcur;
    rhs->next = old_lcur;
}

inline list* min(list* rhs, list* lhs)
{
    return rhs->next->val < lhs->next->val ? rhs : lhs;
}

list* next(list* val, size_t step = 1)
{
    return step == 0 ? val : next(val, step - 1)->next;
}

std::ostream& operator<<(std::ostream& os, list* head)
{
    if (head)
    {
        os << head->val << ' ';
        for (list* beg = head->next; beg != head; beg = beg->next)
        {
            os << beg->val << ' ';
        }
    }
    return os;
}

inline bool is_odd(size_t num)
{
    return num & 1;
}

void heapify(list* beg, list* mid, list* end)
{
    using std::swap;
    while (mid->next != end->next)
    {
        list* max_son = min(beg, beg->next);
        if (max_son->next->val < mid->next->val)
            swap(max_son->next->val, mid->next->val);
        mid = mid->next;
        beg = beg->next->next;
    }
}

void heap_sort(list* beg, list* end, size_t size)
{
    if (size == 1) return;
    using std::swap;
    list* middle = next(beg, size / 2 - 1);
    list* head = beg;
    beg = end;
    end = next(beg, size - 1);
    list* beg2 = beg;
    while (size != 1)
    {
        if (!is_odd(size--))
        {
            if (beg->next->val < middle->next->val)
            {
                swap(beg->next->val, middle->next->val);
            }
            beg2 = beg->next;
            middle = middle->next;
        }
        heapify(beg2, middle, end->next);
        swap(beg->next->val, end->next->val);
        beg2 = beg = beg->next;
    }
}

void bubble_sort(list* before_beg, list* beg, list* end)
{
    list* init_before_beg = before_beg;
    if (before_beg->next->val < beg->next->val)
    {
        swap(before_beg, beg);
        std::swap(before_beg, beg);
        init_before_beg = before_beg;
        if (before_beg->next == end)
            end = beg->next;
    }
    std::cout << beg << '\n';
    while (beg != end)
    {
        if (before_beg->next->val < beg->next->val)
        {
            swap(before_beg, beg);
            std::swap(before_beg, beg);
            if (before_beg->next == end)
                end = beg->next;
        }
        beg = beg->next;
        before_beg = before_beg->next;
    }
    if (init_before_beg != beg->next)
        bubble_sort(init_before_beg, init_before_beg->next, before_beg);
}

void bubble_sort(list* beg, list* end)
{
    list* before_beg = end;
    while (before_beg->next != beg)
        before_beg = before_beg->next;
    bubble_sort(before_beg, beg, end);
}

int main()
{
    size_t nfile_items = count("input.txt");//считаем количество элементов в файле
    std::ifstream ifs { "input.txt" };
    list head { 0, &head };//голова
    bool sort_type;
    ifs >> sort_type;
    ifs >> head.val;
    list* last = &head;
    for (size_t i = 2; i != nfile_items; ++i)
    {
        double tmp;
        ifs >> tmp;
        last = insert(last, tmp);
    }
    if (sort_type == 0)
    {
        heap_sort(&head, last, nfile_items - 1);
    }
    else
    {
        bubble_sort(&head, last);
    }
    ifs.close();
    std::ofstream ofs { "output.txt" };
    //ofs << nfile_items - 1 << ' ';
    //last = next(last, 3);
    //ofs << head.next->val << ' ' << last->next->val << '\n' << head.next->next->val << ' ' << last->next->next->val << '\n';
    //swap(&head, last);
    ofs << &head;
    //ofs << head.next->val << ' ' << last->next->val << '\n' << head.next->next->val << ' ' << last->next->next->val << '\n';
    ofs.close();
    while (erase(&head) != &head);
    return 0;
}