public class b {

    public static void printer(int var) {
        System.out.println("I print " + var);
    }

    public static void printer(double var) {
        System.out.println("I print " + var);
    }

    public static void main(String[] args) {
        Integer var1 = Integer.parseInt("10");
        int var2 = 10;
        double var3 = 10.0;
        char var4 = '1';
        String var5 = "10";

        printer(var1); // I print 10
        printer(var2); // I print 10
        printer(var3); // I print 10.0
        printer(var4); //I print 49
        printer(var5); // The method printer(int) in the type b 
                       //is not applicable for the arguments (String)
    }
}