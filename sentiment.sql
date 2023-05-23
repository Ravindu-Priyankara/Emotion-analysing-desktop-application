create database Sentiment;
use Sentiment;
create table login(id int primary key,username varchar(255) not null,password varchar(255) not null);
create table data(
    id int primary key,
    today date,
    sad int,
    angry int,
    happy int,
    disguest int,
    neutral int,
    fear int,
    suprise int);