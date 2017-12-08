--SQL常用语句汇总
--语句之间一定要用分号隔开
--------------------------------查询------------------------------
select name,imdb_rating from movies;--查询多列数据
select distinct genre from movies; --用于返回唯一不同的值。
select * from movies where name like 'Se_en'; --LIKE 操作符用于在 WHERE 子句中搜索列中的指定模式。此处寻找以Se开头 en结尾的name
select * from movies where name like '%man%';--查找name中包含man
select * from movies where year between 1990 and 2000;
select * from movies where year between 1990 and 2000 and genre='comedy';
select * from movies where genre='comedy' or year<1980;
select * from movies order by imdb_rating desc; --按照imdb_rating降序排列
select * from movies order by imdb_rating Asc limit 3;--查找imdb_rating中最小的三个
--------------------------------函数------------------------
select count(*) from fake_apps;--COUNT(*) 函数返回表中的记录数
select price ,count(*) from fake_apps group by price; --返回每种价格的记录数
select sum(downloads) from fake_apps; --SUM 函数返回数值列的总数（总额）
select max(downloads) from fake_apps;
select avg(downloads) from fake_apps; --AVG 函数返回数值列的平均值。
select price,round(avg(downloads),2) from fake_apps group by price; --round 保留指定小数位数
--------------------------------操作-----------------------------
INSERT INTO celebs (id, name, age) VALUES (5, 'zl', 23); --插入一行
update celebs set age=22 where id=1;--修改表中的数据
alter table celebs add column twitter_handle text;--添加一列
delete from celebs where twitter_handle is NULL; --删除行
create table celeb (id integer primary key,name text unique,date_of_birth text not null,date_of_death text default 'Not Applicable');
--创建表,unique设置该列所有值都不重复,not null设置该列非空,default为该列添加默认值
--创建表时常用的数据类型：integer(size)、int(size)、smallint(size)、tinyint(size)  仅容纳整数。在括号内规定数字的最大位数。
--decimal(size,d)、numeric(size,d) 容纳带有小数的数字。 "size" 规定数字的最大位数。"d" 规定小数点右侧的最大位数。
--char(size) 容纳固定长度的字符串（可容纳字母、数字以及特殊字符）在括号中规定字符串的长度。
--varchar(size) 容纳可变长度的字符串（可容纳字母、数字以及特殊的字符）。在括号中规定字符串的最大长度。
--date(yyyymmdd)  容纳日期。
--------------------------------联结-----------------------------
create table artists(id integer primary key,name text); --id后表示变量类型 primary key 表示该变量作为该表的主键，
                                                         --主键必须包含唯一的值，主键列不能包含 NULL 值
select * from albums where artist_id=3; --条件查询语句
select * from albums join artists on albums.artist_id = artists.id;--内联结查询两张表(返回匹配的行)
select * from albums left join artists on albums.artist_id =artists.id; --即使右表中没有匹配，也从左表返回所有的行
select albums.name as 'album',albums.year,artists.name as 'artist' from albums join artists on albums.artist_id =artists.id where albums.year>1980;
--as 对变量重命名
drop table if exists albums; --当表存在时则删除,不存在时则不进行任何操作
create table if not exists albums (id integer primary key,name text,artist_id integer,year integer,FOREIGN KEY(artist_id) REFERENCES artist(id));
--创建表  FOREIGN KEY 设置外键  一个表中的 FOREIGN KEY 指向另一个表中的 PRIMARY KEY
---------------------------------例题---------------------------------
--以Cno升序、Degree降序查询Score表的所有记录
select * from scores order by cno, degree desc;
--查询Score表中至少有5名学生选修的并以3开头的课程的平均分数(having用于在group by 后增加条件语句)
SELECT Cno,AVG(Degree)FROM ScoresWHERE Cno LIKE '3%'GROUP BY Cno HAVING COUNT(sno) >= 5;
--查询最低分大于70，最高分小于90的Sno列
select sno from scores group by sno having max(degree)<90 and min(degree)
--查询所有学生的Sname、Cname和Degree列  联结三张表 inner join = join
SELECT Sname,Cname,Degree 
FROM Students INNER JOIN Scores 
ON(Students.Sno=Scores.Sno) INNER JOIN Courses
ON(Scores.Cno=Courses.Cno)
ORDER BY Sname;
--查询“95033”班所选课程的平均分(用两个变量分组)
select courses.cname,avg(degree) from students join scores 
on students.sno=scores.sno join courses
on courses.cno=scores.cno where class='95033' --where语句要写在group by 前面
group by courses.cno,courses.cname;


















