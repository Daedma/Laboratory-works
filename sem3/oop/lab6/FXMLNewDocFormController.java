import java.util.ResourceBundle;

import gui.Controller;
import java.net.URL;
import javafx.event.ActionEvent;
import javafx.event.EventHandler;
import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.fxml.Initializable;
import javafx.scene.Node;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.scene.control.Alert;
import javafx.scene.control.Dialog;
import javafx.scene.control.Spinner;
import javafx.scene.control.SpinnerValueFactory;
import javafx.scene.control.TextField;
import javafx.scene.control.Alert.AlertType;
import javafx.stage.Modality;
import javafx.stage.Stage;
import javafx.stage.WindowEvent;

public class FXMLNewDocFormController implements Controller, Initializable {
	private static final int OK = 0;

	private static final int CANCEL = 1;

	private int lastButton = OK;

	private Stage stage;

	private static FXMLNewDocFormController dialogWindow = null; // Singleton

	@FXML
	private TextField edRightBorder = new TextField();

	@FXML
	private TextField edLeftBorder = new TextField();

	@FXML
	private Spinner<Integer> edPointsCount = new Spinner<Integer>();

	public double getLeftDomainBorder() {
		try {
			return Double.parseDouble(edLeftBorder.getText());
		} catch (Exception e) {
		}
		return Double.NaN;
	}

	public double getRightDomainBorder() {
		try {
			return Double.parseDouble(edRightBorder.getText());
		} catch (Exception e) {
		}
		return Double.NaN;
	}

	public int getPointsCount() {
		return edPointsCount.getValue();
	}

	public void redraw() {

	}

	public void setStage(Stage stage) {
		this.stage = stage;
		stage.setOnCloseRequest(new EventHandler<WindowEvent>() {
			public void handle(WindowEvent event) {
				lastButton = CANCEL;
				stage.hide();
			}
		});
	}

	@FXML
	private void confirm(ActionEvent ae) {
		try {
			FunctionGUIApp.tabFDoc.newFunction(Double.parseDouble(edLeftBorder.getText()),
					Double.parseDouble(edRightBorder.getText()), edPointsCount.getValue());
			lastButton = OK;
			stage.hide();
		} catch (Exception e) {
			Alert errorMessage = new Alert(AlertType.ERROR);
			errorMessage.setHeaderText("Error");
			errorMessage.setContentText(e.getLocalizedMessage());
			errorMessage.setResizable(false);
			errorMessage.showAndWait();
		}

	}

	@FXML
	private void cancel(ActionEvent ae) {
		lastButton = CANCEL;
		stage.hide();
	}

	public void initialize(URL location, ResourceBundle resources) {
		SpinnerValueFactory<Integer> valueFactory = new SpinnerValueFactory.IntegerSpinnerValueFactory(2,
				Integer.MAX_VALUE);
		edPointsCount.setValueFactory(valueFactory);
		edPointsCount.setEditable(true);
		edLeftBorder.setText("0");
		edRightBorder.setText("10");
	}

	public static int showDialog(Stage primaryStage) {
		if (dialogWindow == null) {
			try {
				Stage stage = new Stage();
				FXMLLoader loader = new FXMLLoader(FXMLNewDocFormController.class.getResource("FXMLNewDocForm.fxml"));
				Parent root = loader.load();
				dialogWindow = loader.getController();
				dialogWindow.setStage(stage);
				Scene scene = new Scene(root);
				stage.setTitle("Function parameters");
				stage.setResizable(false);
				stage.setScene(scene);
				stage.initModality(Modality.APPLICATION_MODAL);
				stage.initOwner(primaryStage);
				stage.showAndWait();
			} catch (Exception e) {
				System.err.println(e.getClass().getSimpleName() + " : " + e.getLocalizedMessage());
			}
		} else {
			dialogWindow.stage.show();
		}
		return dialogWindow.lastButton;
	}
}
