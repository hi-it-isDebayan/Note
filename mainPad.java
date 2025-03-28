
import java.io.*;
import java.util.*;

public class mainPad {
    public static void storeTitle(String title, String path, String mainPath) throws IOException {
        Date dt = new Date();
        FileWriter mainW = new FileWriter(mainPath, true); // Append mode enabled
        String set = dt + "\t" + title + "\tlocation = " + path + "\n";

        mainW.write(set);
        mainW.close();
    }

    public static File createFile(String path) throws IOException {
        File fp = new File(path);
        if (fp.createNewFile()) {
            System.out.println("New file created: " + path);
        } else {
            System.out.println("File already exists: " + path);
        }
        return fp;
    }

    public static void main(String[] args) throws IOException {
        Scanner conin = new Scanner(System.in);
        System.out.print("Title: ");
        String title = conin.nextLine();

        File lastVisit = createFile("storage\\Java_prob\\Notepad\\lastVisitedInfo.txt");
        File f = createFile("storage\\Java_prob\\Notepad\\" + title + ".txt");

        storeTitle(title, f.getAbsolutePath(), lastVisit.getAbsolutePath());
    }
}
