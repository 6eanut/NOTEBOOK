# what is nohup?

**nohup:**

no hangup

**function:**

A command used in Unix-like operating systems to keep a process running after the user quits or the session is disconnected. By default, when the user exits a terminal session, the terminal sends a SIGHUP signal to all processes associated with that session, which usually results in the termination of those processes. The nohup command prevents this.

**usage**:

```
//Simple
nohup mycommand
//Specified output file
nohup mycommand > output.output
//Plus standard error
nohup mycommand > output.output 2>&1
//Background run
nohup mycommand &
```

**top:**

```
//Viewing Process Information
top
```

**kill:**

```
kill -9 process_id
```
