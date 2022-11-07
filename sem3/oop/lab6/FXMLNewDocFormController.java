import java.util.ResourceBundle;

import gui.Controller;
import java.net.URL;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.control.Alert;
import javafx.scene.control.Spinner;
import javafx.scene.control.TextField;
import javafx.scene.control.Alert.AlertType;

public class FXMLNewDocFormController implements Controller, Initializable {
	@FXML
	private TextField edRightBorder;

	@FXML
	private TextField edLeftBorder;

	@FXML
	private Spinner<Integer> edPointsCount;

	public void redraw() {

	}

	@FXML
	private void confirm(ActionEvent ae) {
		try {
			FunctionGUIApp.tabFDoc.newFunction(Double.parseDouble(edLeftBorder.getText()),
					Double.parseDouble(edRightBorder.getText()), edPointsCount.getValue());

		} catch (Exception e) {
			Alert errorMessage = new Alert(AlertType.ERROR);
			errorMessage.setHeaderText("Error");
			errorMessage.setContentText(e.getLocalizedMessage());
			errorMessage.showAndWait();
		}

	}

	@FXML
	private void cancel(ActionEvent ae) {

	}

	public void initialize(URL location, ResourceBundle resources) {

	}
}
