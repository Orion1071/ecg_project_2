import java.util.HashMap;
import java.util.Hashtable;
import java.util.ArrayList;
import javax.naming.directory.SearchResult;

public class b {

    public static void printer(int var) {
        System.out.println("I print " + var);
    }

    public static void printer(double var) {
        System.out.println("I print " + var);
    }

    public static void multi_data_type(ArrayList<SearchResult<T>> arr_in, Hashtable<F, HashMap<A, B>>map_in) {
        // IMPLEMENT
    }
    public static void main(String[] args) {
        Integer var_Int = Integer.parseInt("10");
        int var_int = 10;
        double var_double = 10.0;
        Double var_Double = 11.0;
        char var_char = '1';
        float var_float = 12.0f;
        String var_str = "10";
        

        printer(var_Int); // I print 10
        printer(var_int); // I print 10
        printer(var_double); // I print 10.0
        printer(var_Double); //I print 11.0
        printer(var_char); // I print 49
        printer(var_float); // I print 12.0
        // printer(var_str); // The method printer(int) in the type b 
                       //is not applicable for the arguments (String)


        ArrayList<Integer> arr_old = new ArrayList<>();
        ArrayList<SearchResult<Integer>> arr_new = new ArrayList<>();
        HashMap<Integer, String> map = new HashMap<>(); 
        Hashtable<Integer,String> hash_table_old = new Hashtable<>();
        Hashtable<Integer, HashMap<Integer, String>> hash_table_new = new Hashtable<>();

        multi_data_type(arr_old, hash_table_old);
        multi_data_type(arr_new, hash_table_new);



    }




}
