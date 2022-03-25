#include <fstream>
#include <utility>
#include <algorithm>

enum class Sort_mode { HEAP, BUBBLE };//Способ сортировки

//Односвязный список
struct list
{
    double val;
    list* next;//указатель на следующий элемент в списке
};

//вставка элемента
list* insert_after(list* pos, double val)
{
    return pos->next = new list { val, pos->next };
}

//удаление элемента после _lst
list* erase_after(list* pos)
{
    list* next = pos->next->next;//сохраняем указатель на следующий за удаляемым элементом
    delete pos->next;//удаляем элемент
    return pos->next = next;//текущий элемент теперь должен ссылаться на следующий за удаленным
}

//очищение списка
void clear(list*& head)
{
    while (erase_after(head) != head);
    delete head;
    head = nullptr;
}

//меняет местами два элемента, находящихся за переданными 
void swap(list* lhs, list* rhs)
{
    //два случая для соседних элементов
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
    //перекидываем ссылки
    list* old_rcur = rhs->next;
    list* old_lcur = lhs->next;
    list* old_lnext = lhs->next->next;
    lhs->next->next = rhs->next->next;
    rhs->next->next = old_lnext;
    lhs->next = old_rcur;
    rhs->next = old_lcur;
}

//возвращает указатель на минимальный элемент из переданных
inline list* min(list* rhs, list* lhs)
{
    return rhs->val < lhs->val ? rhs : lhs;
}

//возвращает указатель на элемент, находящийся в нескольких шагах от переданного
list* next(list* val, size_t step = 1)
{
    return step == 0 ? val : next(val, step - 1)->next;
}

//возвращает указатель на предыдущий элемент
list* prev(list* p)
{
    list* res = p;
    while (res->next != p) res = res->next;
    return res;
}

//вывод элементов списка через пробел
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

//проверка числа на нечетность
inline bool is_odd(size_t num)
{
    return num & 1;
}

//создание бинарного сортирующего дерева
void heapify(list* beg, list* mid, list* end)
{
    while (mid != end)
    {
        list* max_son = min(beg, beg->next);
        if (max_son->val < mid->val)
            std::swap(max_son->val, mid->val);
        mid = mid->next;
        beg = beg->next->next;
    }
}

//пирамидальная сортировка
void heap_sort(list* beg, list* end, size_t size)
{
    if (size == 1) return;//если размер равен единице, то не сортируем
    list* middle = next(beg, size / 2);
    list* beg2 = beg;
    while (size != 1)
    {
        if (!is_odd(size--))//отдельная обработка для случая, когда размер сортируемогого подсписка четен
        {
            if (beg->val < middle->val)
            {
                std::swap(beg->val, middle->val);
            }
            beg2 = beg->next;
            middle = middle->next;
        }
        heapify(beg2, middle, end->next);
        std::swap(beg->val, end->val);
        beg2 = beg = beg->next;
    }
}

//сортировка пузырьком
void bubble_sort(list*& first, list* end)
{
    if (first->next == first) return;//если в списке один элемент, то не сортируем
    list* beg = end;
    end = prev(end);
    first = beg;
    while (beg != end)
    {
        while (beg->next != end)
        {
            if (beg->next->next->val < beg->next->val)
            {
                if (beg->next == first)
                    first = beg->next->next;
                if (beg->next->next == end)
                    end = beg->next;
                swap(beg, beg->next);//используем swap с перекидыванием ссылок
            }
            beg = beg->next;
        }
        if (end->next->val < end->val)
        {
            if (end->next == first)
                first = end;
            swap(beg, beg->next);
        }
        end = beg;
        beg = first;
    }
    first = first->next;
}

int main()
{
    list* head = new list;//создаем первый элемент списка
    head->next = head;//зацикливаем список
    int sort_type;//тип сортировки
    std::ifstream ifs { "input.txt" };//открываем файловый поток
    ifs >> sort_type;//считываем тип сортировки
    ifs >> head->val;//считываем первый элемент
    list* last = head;//указатель на последний элемент списка
    size_t list_size = 1;//размер списка
    for (double tmp; ifs >> tmp; last = insert_after(last, tmp), ++list_size);//заполняем список
    if (sort_type == static_cast<int>(Sort_mode::HEAP))
    {
        heap_sort(head, last, list_size);
    }
    else
    {
        bubble_sort(head, last);
    }
    ifs.close();
    std::ofstream ofs { "output.txt" };
    ofs << list_size << ' ' << head;//выводим результат сортировки
    ofs.close();
    clear(head);//очищаем список
    return 0;
}