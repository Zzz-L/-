## 1）目录操作：ls、cd、mkdir、find、locate、whereis等
- ls: 查看所有文件夹  
- cd：到达指定目录  
- mkdir：创建目录  
- find:  指定目录搜索（默认当前目录），可以使用文件的属性和权限进行搜索  
- locate: 快速查找系统数据库中指定的内容。  
- whereis: 速度比较快，因为它只搜索几个特定的目录。   
## 2）文件操作：mv、cp、rm、touch、cat、more、less
- mv: 移动文件，更改文件名  
　　-i: 交互模式，如果目标文件存在，则询问是否覆盖  
　　-r: 移动目录，跟改目录名  
- cp: 拷贝文件 目录  
　　-i: 交互模式，如果目标文件存在，则询问是否覆盖  
　　-r: 拷贝目录  
　　cp file1 file2 file3.... dir 表示将file1,file2...拷贝到dir  
　　cp -r dir1 dir2 dir3... dirn 将dir1, dir2,dir3...拷贝到dirn  
- rm: 删除文件  
　　-i: 交互模式，询问是否删除  
　　rm -r dir1 dir2 dir3...可删除多个   
- rmdir: 删除目录
- cat: 显示指定文件的全部内容  
- more: 分页显示指定文件内容  
- Less：用分页的形式显示指定文件的内容，区别是more和less翻页使用的操作键不同。    
- head: 取出前面几行 
- tail: 取出后面几行
- grep:查找文件里符合条件的字符串，并把匹配的行打印出来，可以使用正则表达式。 
- tar: 打包压缩  
     -c  归档文件  
     -x  压缩文件  
     -z  gzip压缩文件  
     -j  bzip2压缩文件  
     -v  显示压缩或解压缩过程 v(view)  
     -f  使用档名  
例：
tar -cvf /home/abc.tar /home/abc              只打包，不压缩  
tar -zcvf /home/abc.tar.gz /home/abc        打包，并用gzip压缩  
tar -jcvf /home/abc.tar.bz2 /home/abc      打包，并用bzip2压缩  
当然，如果想解压缩，就直接替换上面的命令  tar -cvf  / tar -zcvf  / tar -jcvf 中的“c” 换成“x” 就可以了。  
- Zip/unzip:压缩解压缩.zip文件   
- wc: 计数　　
　　-l: 行数　　
　　-w: 字数　　
　　-c: 字符数　　
　　wc -l file1 file2 ......可以统计多个文件　　
- touch: 创建空文件
- echo: 创建带有内容的文件。
## 3）权限操作：chmod+rwx421
chmod [u所属用户  g所属组  o其他用户  a所有用户]  [+增加权限  -减少权限]  [r  w  x]   目录名   
chmod u+x g+w o+r  filename
chmod u+x file ：给file的属主增加执行权限  
R: 读 数值表示为4  
W: 写  数值表示为2  
X: 可执行 数值表示为1  
## 4）账号操作：su、whoami、last、who、w、id等
- Su：切换用户命令  
- Sudo：一系统管理员的身份执行命令  
- Passwd：用于修改用户的密码  
- Who：显示系统中有那些用户在使用。  
　　　-ami  显示当前用户   
　　　-u：显示使用者的动作/工作   
　　　-s：使用简短的格式来显示   
　　　-v：显示程序版本   
- Last：显示每月登陆系统的用户信息  
## 5）查看系统：history、top
history： 查看历史命令  
Top: 动态地显示进程  
PS: 查看当前进程  
kill: 终止进程
## 6）关机重启：shutdown、reboot
Shutdown：-r 关机后立即重启    
　　　　　-k 并不真正的关机，而只是发出警告信息给所有用户   
　　　　　-h 关机后不重新启动    
Reboot： 用于计算机重启   
## 7）vim操作：i、w、w!、q、q!、wq等
　　选中：v
　　复制：y
　　粘贴：p
　　删除：d
## 8) 其他
man 帮助命令，查看命令详解,只要觉得哪个命令不清楚，man它就可以了.  
热键“中断目前程序”：Ctrl+C
