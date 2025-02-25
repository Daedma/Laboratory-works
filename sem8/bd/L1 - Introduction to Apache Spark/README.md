# Задание

[L1_Apache_Spark.md](L1_Apache_Spark.md)


# Развёртывание MapR с помощью Docker

https://hub.docker.com/r/maprtech/dev-sandbox-container

```
sudo rm -rf /tmp/maprdemo

sudo mkdir -p /tmp/maprdemo/hive /tmp/maprdemo/zkdata /tmp/maprdemo/pid /tmp/maprdemo/logs /tmp/maprdemo/nfs
sudo chmod -R 777 /tmp/maprdemo/hive /tmp/maprdemo/zkdata /tmp/maprdemo/pid /tmp/maprdemo/logs /tmp/maprdemo/nfs

export clusterName="maprdemo.mapr.io"
export MAPR_EXTERNAL='0.0.0.0'
export PORTS='-p 9998:9998 -p 8042:8042 -p 8888:8888 -p 8088:8088 -p 9997:9997 -p 10001:10001 -p 8190:8190 -p 8243:8243 -p 2222:22 -p 4040:4040 -p 7221:7221 -p 8090:8090 -p 5660:5660 -p 8443:8443 -p 19888:19888 -p 50060:50060 -p 18080:18080 -p 8032:8032 -p 14000:14000 -p 19890:19890 -p 10000:10000 -p 11443:11443 -p 12000:12000 -p 8081:8081 -p 8002:8002 -p 8080:8080 -p 31010:31010 -p 8044:8044 -p 8047:8047 -p 11000:11000 -p 2049:2049 -p 8188:8188 -p 7077:7077 -p 7222:7222 -p 5181:5181 -p 5661:5661 -p 5692:5692 -p 5724:5724 -p 5756:5756 -p 10020:10020 -p 50000-50050:50000-50050 -p 9001:9001 -p 5693:5693 -p 9002:9002 -p 31011:31011'

docker run --name maprdemo -d --privileged -v /tmp/maprdemo/zkdata:/opt/mapr/zkdata -v /tmp/maprdemo/pid:/opt/mapr/pid  -v /tmp/maprdemo/logs:/opt/mapr/logs  -v /tmp/maprdemo/nfs:/mapr $PORTS -e MAPR_EXTERNAL -e clusterName -e isSecure --hostname ${clusterName} maprtech/dev-sandbox-container:latest > /dev/null 2>&1

sleep 400

docker exec -d maprdemo usermod -s /bin/bash root
docker exec -d maprdemo usermod -s /bin/bash mapr
docker exec -d maprdemo apt install -y mapr-resourcemanager mapr-nodemanager mapr-historyserver
docker exec -d maprdemo /opt/mapr/server/configure.sh -R

```

Код взят из инструкции https://docs.datafabric.hpe.com/62/MapRContainerDevelopers/RunMapRContainerDevelopers.html.   
Протестирован в WSL2 Ubuntu 18.04 https://docs.microsoft.com/ru-ru/windows/wsl/install-win10.

Время развёртывания: ~10 минут.

## Подключение к контейнеру по ssh

```
ssh root@localhost -p 2222
```
или
```
ssh mapr@localhost -p 2222
```

Пароль: mapr


## Настройка Spark

Добавьте расположение Spark утилит в переменную среды PATH для их вызова без указания полного пути.
```
echo 'export PATH=$PATH:/opt/mapr/spark/spark-2.4.5/bin' > /home/mapr/.bash_profile
```

https://docs.datafabric.hpe.com/62/AdvancedInstallation/InstallingHadoop.html

## Веб-интерфейсы

| Имя сервиса | Адрес | Комментарии |
|---|---|---|
| CLDB | localhost:7221  | Должен запуститься в первую очередь. |
| Веб консоль администратора MapR | localhost:8443  | Используется самоподписанный сертификат - сделайте исключение для этого сайта. |
| Resource manager | localhost:8088  | |
| History server | localhost:19888 | |

https://docs.datafabric.hpe.com/62/ReferenceGuide/MapRPorts.html

Актуальные адреса сервисов можно узнать в консоли администратора.
