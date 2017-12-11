---------------------------------例题---------------------------------
--1.以Cno升序、Degree降序查询Score表的所有记录
select * from scores order by cno, degree desc;
--2.查询Score表中至少有5名学生选修的并以3开头的课程的平均分数(having用于在group by 后增加条件语句)
SELECT Cno,AVG(Degree)FROM ScoresWHERE Cno LIKE '3%'GROUP BY Cno HAVING COUNT(sno) >= 5;
--3.查询最低分大于70，最高分小于90的Sno列
select sno from scores group by sno having max(degree)<90 and min(degree)
--4.查询所有学生的Sname、Cname和Degree列  联结三张表 inner join = join
SELECT Sname,Cname,Degree 
FROM Students INNER JOIN Scores 
ON(Students.Sno=Scores.Sno) INNER JOIN Courses
ON(Scores.Cno=Courses.Cno)
ORDER BY Sname;
--5.查询“95033”班所选课程的平均分(用两个变量分组)
select courses.cname,avg(degree) from students join scores 
on students.sno=scores.sno join courses
on courses.cno=scores.cno where class='95033' --where语句要写在group by 前面
group by courses.cno,courses.cname;
--6.查询所有同学的Sno、Cno和rank列
+-----+-------+--------+      
| sno | cno   | degree |           
+-----+-------+--------+           
| 103 | 3-245 |   86.0 |                
| 105 | 3-245 |   75.0 |
| 109 | 3-245 |   68.0 |                                  
| 103 | 3-105 |   92.0 |                  
| 105 | 3-105 |   88.0 |
| 109 | 3-105 |   76.0 |
| 101 | 3-105 |   64.0 |
| 107 | 3-105 |   91.0 |
| 108 | 3-105 |   78.0 |
| 101 | 6-166 |   85.0 |
| 107 | 6-106 |   79.0 |
| 108 | 6-166 |   81.0 |
+-----+-------+--------+
+------+------+------+
| low  | upp  | rank |
+------+------+------+
|   90 |  100 | A    |
|   80 |   89 | B    |
|   70 |   79 | C    |
|   60 |   69 | D    |
|    0 |   59 | E    |
+------+------+------+
--即使两张表没有确定列联结,但也可以根据相关关系进行联结
SELECT Sno,Cno,rank FROM Scores INNER JOIN grade ON (Scores.Degree>=grade.low AND Scores.Degree<=grade.upp) ORDER BY Sno;    
--8.查询成绩高于学号为“109”、课程号为“3-105”的成绩的所有记录
SELECT s1.Sno,s1.Degree
FROM Scores AS s1 INNER JOIN Scores AS s2
ON(s1.Cno=s2.Cno AND s1.Degree>s2.Degree)
WHERE s1.Cno='3-105' AND s2.Sno='109'
ORDER BY s1.Sno;--等价于
select * from scores where cno='3-105'
and degree>(select degree from scores
where sno='109' and cno='3-105');
--9.查询和学号为108的同学同年出生的所有学生的Sno、Sname和Sbirthday列
select sno,sname,sbirthday from students
where year(sbirthday)=(select year(sbirthday) --year()提取出时间格式中的年份
from students where sno='108'); --等价于  
SELECT s1.Sno,s1.Sname,s1.Sbirthday 
FROM Students AS s1 INNER JOIN Students AS s2
ON(YEAR(s1.Sbirthday)=YEAR(s2.Sbirthday))
WHERE s2.Sno='108';
--10.查询选修某课程的同学人数多于5人的教师姓名
select distinct tname from teachers join courses
 on courses.tno=teachers.tno join scores
on scores.cno=courses.cno where courses.cno = --需指明此处cno属于哪张表
(select cno from scores group by
cno having count(*) >5);
--11.查询出“计算机系“教师所教课程的成绩表
select tname,cname,sname,degree from scores join courses --即使不是来自该表的变量也可写出
on courses.cno=scores.cno join teachers --只要联结其他表,所有变量都可输出
on teachers.tno=courses.tno join students
on scores.sno=students.sno where
teachers.depart='计算机系' order by tname,cname,degree desc;
--12.查询选修编号为“3-105“课程且成绩至少高于任意选修编号为“3-245”的同学的成绩的Cno、Sno和Degree,并按Degree从高到低次序排序
select sno,cno,degree from scores
where cno='3-105' and degree >= 
(select min(degree) from scores
where cno='3-245') order by degree desc; --等价于
SELECT Cno,Sno,Degree
FROM Scores
WHERE Cno='3-105' AND Degree > ANY(  --any表示任意选修
    SELECT Degree
    FROM Scores
    WHERE Cno='3-245')
ORDER BY Degree DESC;
--13.查询所有教师和同学的name、sex和birthday
SELECT Sname,Ssex,Sbirthday
FROM Students
UNION  --UNION 操作符用于合并两个或多个 SELECT 语句的结果集
SELECT Tname,Tsex,Tbirthday
FROM Teachers;
--14.查询成绩比该课程平均成绩低的同学的成绩表
select s1.* from scores as s1 join (
select cno,avg(degree) as adegree from scores  --计算课程平均成绩表,并与原始成绩表联结
group by cno) s2 on (s1.cno=s2.cno and s1.degree<s2.adegree); --通过此方法查询成绩低于平均成绩的同学
--15.查询至少有2名男生的班号
select class,count(*) as boycount from
students where ssex='男' group by class --通过where语句筛选性别
having boycount>=2;
--16.查询Student表中每个学生的姓名和年龄
SELECT Sname,YEAR(NOW())-YEAR(Sbirthday) AS Sage --now() 返回现在的时间
FROM Students;
--17.查询和“李军”同性别并同班的同学Sname
select s1.sname from students as s1  --联结同一张表
join students as s2 on (s1.ssex=s2.ssex
and s1.class=s2.class) where s2.sname='李军';

