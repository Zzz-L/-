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
--18.从titles表获取按照title进行分组，每组个数大于等于2，给出title以及对应的数目t,注意对于重复的emp_no进行忽略。
select title,count(distinct emp_no) as t from titles group by title having t>=2;  --count函数里面可添加其他参数
--19.查找employees表所有emp_no为奇数，且last_name不为Mary的员工信息，并按照hire_date逆序排列
select * from employees where last_name!='Mary' and emp_no % 2=1 order by hire_date desc; --奇数表明取余为1
--20.查找所有员工自入职以来的薪水涨幅情况，给出员工编号emp_noy以及其对应的薪水涨幅growth，并按照growth进行升序
--本题思路是先分别用两次LEFT JOIN左连接employees与salaries，建立两张表，分别存放员工当前工资（sCurrent）与员工
--入职时的工资（sStart），再用INNER JOIN连接sCurrent与sStart，最后限定在同一员工下用当前工资减去入职工资
SELECT sCurrent.emp_no, (sCurrent.salary-sStart.salary) AS growth  --直接用where语句联结两张表更简洁
FROM (SELECT s.emp_no, s.salary FROM employees e, salaries s WHERE e.emp_no = s.emp_no AND s.to_date = '9999-01-01') AS sCurrent,
(SELECT s.emp_no, s.salary FROM employees e, salaries s WHERE e.emp_no = s.emp_no AND s.from_date = e.hire_date) AS sStart
WHERE sCurrent.emp_no = sStart.emp_no
ORDER BY growth
--21.查询学过“001”并且也学过编号“002”课程的同学的学号、姓名；
--解法一：求交集
select s2.sno,s1.sname from student s1,sc s2
where s1.sno=s2.sno and s2.cno='001'
intersect   --获取两张表返回结果的交集
select s2.sno,s1.sname from student s1,sc s2
where s1.sno=s2.sno and s2.cno='002';
--解法二：利用exists (exist用法类似于in,但In引导的子句只能返回一个字段)
select Student.S#,Student.Sname from Student,SC where Student.S#=SC.S# and SC.C#='001'
and exists( Select * from SC as SC_2 where SC_2.S#=SC.S# and SC_2.C#='002'); 

