import docs.TabulatedFunctionDoc;
import javafx.application.Application;
import javafx.stage.*;
import javafx.fxml.*;
import javafx.scene.*;

public class FunctionGUIApp extends Application {
	public static void main(String[] args) {
		launch(args);
	}

	public static TabulatedFunctionDoc tabFDoc;

	@Override
	public void start(Stage stage) throws Exception {
		FXMLLoader loader = new FXMLLoader(getClass().getResource("FXMLMainForm.fxml"));
		Parent root = loader.load();
		FXMLMainFormController ctrl = loader.getController();
		Scene scene = new Scene(root);
		stage.setTitle("Впишите сюда заголовок главного окна");
		stage.setScene(scene);
		stage.show();
	}

	public void init() throws Exception {
		super.init();
		if (tabFDoc == null) {
			tabFDoc = new TabulatedFunctionDoc();
			tabFDoc.newFunction(0, 100000, 1000);
		}
	}

}
