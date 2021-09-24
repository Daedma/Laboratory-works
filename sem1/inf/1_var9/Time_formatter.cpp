/*
Программа должна переводить количество секунд, прошедших с 
начала суток, в формат «часы:минуты:секунды». 
Входные  данные. Целое  неотрицательное  число –  количество 
секунд, прошедших с начала суток. 
Выходные данные. Количество часов, прошедших с начала су-
ток, количество минут, прошедших с начала последнего часа, и ко-
личество секунд, прошедших с начала последней минуты, в формате 
«часы:минуты:секунды».
*/
#include <iostream>
#include <sstream>
#include <limits>

class Time //Класс, представляющий время, с необходимым для решения задачи функционалом
{
public:
    void add_seconds(uint64_t) noexcept; //метод для добавления секунд к началу суток (00:00:00)
    std::string format() const noexcept; //метод для приведения времени к отформатированной строке

private:
    //все значения, принимаемые переменными, лежат в отрезке [0; 59], поэтому нет смысла использовать большие типы
    uint16_t Hour_ = 0U;   //Часы
    uint16_t Minute_ = 0U; //Минуты
    uint16_t Second_ = 0U; //Секунды
};

void Time::add_seconds(uint64_t sec) noexcept
{
    Second_ = sec % 60ULL;           //в минуте 60 секунд
    Minute_ = (sec / 60ULL) % 60ULL; //в скобках получаем количество минут, после отсекаем то, что останется в минутах
    Hour_ = (sec / 3600ULL) % 24ULL; //в скобках получаем количество часов и оставляем то, что умещается в день
}

//очень маловероятно, что кинется std::bad_alloc за пределы метода из-за нехватки памяти, потому что в нашей строке всего 8 символов
//и, вероятно, память на куче выделяться и вовсе не будет (SSO)
std::string Time::format() const noexcept
{
    std::ostringstream oss;
    if (Hour_ < 10U)     //если число однозначное,
        oss << '0';      //то добавить один незначащий ноль, чтобы привести время к требуемому формату
    oss << Hour_ << ':'; //записать в строку количество часов и разделитель
    //аналогично поступаем с минутами и секундами...
    if (Minute_ < 10U)
        oss << '0';
    oss << Minute_ << ':';
    if (Second_ < 10U)
        oss << '0';
    oss << Second_;
    return oss.str(); //вернём копию записанной строки
}

std::ostream &operator<<(std::ostream &os, const Time &time)
{
    os << time.format();
    return os;
}

int main()
{
    std::cout << "Time formatter" << std::endl; //название программы
    char choice;                                //выбор пользователя о завершении работы программы или продолжении её выполнения
    do
    {
        std::cout << "Input the number of seconds>"; //приглашение к вводу
        int64_t seconds{0};
        while (!(std::cin >> seconds) || seconds < 0 || std::cin.peek() != '\n') //пока ввод не будет успешным
        {
            std::cin.clear();                                                   //сбросим флаги ошибок
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); //очистим буфер введённых пользователем значений
            if (seconds < 0 || seconds == std::numeric_limits<int64_t>::max())  //если число отрицательное или слишком большое, то ...
                std::cout << "The number must belong to the segment [0; "
                          << std::numeric_limits<int64_t>::max() << "]. Please try again>"; //... сообщить об этом пользователю
            else
                std::cout << "Incorrect input! Please try again>"; //приглашение к повторному вводу
        }
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); //очистим буфер введённых пользователем значений
        Time time{};
        time.add_seconds(seconds);               //добавим секунды к началу суток
        std::cout << time.format() << std::endl; //выведем результат
        std::cout << "Continue? (Y/N)>";         //узнаем, желает ли пользователь продолжить или же он желает завершить работу программы.
        std::cin >> choice;
        while (choice != 'Y' && choice != 'N' || std::cin.peek() != '\n') //пока не будут переданы правильные значения в choice
        {
            std::cout << "Incorrect input! Please try again>";                  //приглашение к повторному вводу
            std::cin.clear();                                                   //сбросим флаги ошибок
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); //очистим буфер
            std::cin >> choice;
        }
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); //очистим буфер
    } while (choice == 'Y');                                                //пока пользователь желает продолжать пользоваться этой увлекательной и полезной программой
    std::cout << "Good bye!\n";                                             //прощаемся с пользователем
    return 0;                                                               //программа отработала корректно
}
