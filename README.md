## Changed Methods Finder (test task)

This script finds changed methods between two last commits in local repository.
To run the script, type python get_java_methods.py "path or nothing"
All changed methods will be printed and marked as "changed", "deleted" or "added".

Some not obvious rules:
1. If name of the class, where method is, have been changed - old and new methods are marked as "deleted" and "added".
2. If name of the method have been changed -  old and new methods are marked as "deleted" and "added".

### Example
For my local clone of [repository](https://github.com/lissrbay/uni_proj) located in C:\Users\lissrbay\Desktop\bugml\uni_proj, we can run script by command:

python get_java_methods.py "C:\Users\lissrbay\Desktop\bugml\uni_proj"

Result:

{'public static int hash32(int sum, final String string) - changed', 'public static string hash32(int sum, final String string) - deleted'}

### Used libraries

* GitPython==3.1.8
