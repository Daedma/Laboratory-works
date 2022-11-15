
import functions.FunctionPoint;
import gui.Controller;
import gui.FunctionPointT;
import javafx.event.ActionEvent;
import javafx.event.EventHandler;
import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.fxml.Initializable;
import javafx.scene.Parent;
import javafx.scene.control.Alert;
import javafx.scene.control.Button;
import javafx.scene.control.ButtonType;
import javafx.scene.control.Label;
import javafx.scene.control.Menu;
import javafx.scene.control.MenuBar;
import javafx.scene.control.TableColumn;
import javafx.scene.control.TableRow;
import javafx.scene.control.TableView;
import javafx.scene.control.TextField;
import javafx.scene.control.ToolBar;
import javafx.scene.control.Alert.AlertType;
import javafx.scene.control.ButtonBar.ButtonData;
import javafx.scene.control.cell.PropertyValueFactory;
import javafx.scene.input.MouseButton;
import javafx.scene.input.MouseEvent;
import javafx.stage.Stage;
import javafx.stage.WindowEvent;
import javafx.stage.FileChooser.ExtensionFilter;
import javafx.application.Application;
import javafx.collections.ObservableList;
import javafx.stage.FileChooser;
import javafx.stage.Modality;
import javafx.scene.*;

import java.io.File;
import java.net.URL;
import java.util.Comparator;
import java.util.Optional;
import java.util.ResourceBundle;

public class FXMLMainFormController implements Initializable, Controller {
	@FXML
	private TextField edY;

	@FXML
	private TextField edX;

	@FXML
	private Label labelPointNumber;

	@FXML
	private TableView<FunctionPointT> table = new TableView<FunctionPointT>();

	@FXML
	private TableColumn<FunctionPointT, Double> columnX = new TableColumn<FunctionPointT, Double>("X values");

	@FXML
	private TableColumn<FunctionPointT, Double> columnY = new TableColumn<FunctionPointT, Double>("Y values");

	@FXML
	private Button buttonAddPoint;

	@FXML
	private Button buttonDelete;

	@FXML
	private Label labelTextFieldX;

	@FXML
	private Label labelTextFieldY;

	@FXML
	private MenuBar menuBar;

	@FXML
	private Menu menuFile;

	@FXML
	private Menu menuTabulate;

	private Stage primaryStage;

	private Stage dialogStage;

	private FXMLNewDocFormController dialogController;

	public void setStage(Stage stage) {
		primaryStage = stage;
		primaryStage.setOnCloseRequest(new EventHandler<WindowEvent>() {
			public void handle(WindowEvent we) {
				if (FunctionGUIApp.tabFDoc.isModified()) {
					ButtonType yes = new ButtonType("Yes", ButtonData.YES);
					ButtonType no = new ButtonType("No", ButtonData.NO);
					Alert alert = new Alert(AlertType.CONFIRMATION,
							"Your document has unsaved changes!\nDo you want to save it?", yes, no);
					alert.setTitle("Tabulated functions");
					// alert.setContentText();
					Optional<ButtonType> result = alert.showAndWait();
					if (result.isPresent()) {
						if (result.get() == ButtonType.YES)
							saveFile(null);
						else
							System.exit(0);
					}
				} else
					System.exit(0);
			}
		});
	}

	// Events

	@FXML
	private void openFile(ActionEvent av) {
		FileChooser fileChooser = new FileChooser();
		fileChooser.setTitle("Open tabulated function document");
		fileChooser.getExtensionFilters().add(new ExtensionFilter("JSON Document (*.json)", "*.json"));
		fileChooser.setInitialDirectory(new File(".\\"));
		File file = fileChooser.showOpenDialog(primaryStage);
		if (file == null)
			return;
		try {
			FunctionGUIApp.tabFDoc.loadFunction(file.getAbsolutePath());
		} catch (Throwable e) {
			showErrorMessage("Failed to load function!");
		}
	}

	@FXML
	private void saveAsFile(ActionEvent av) {
		FileChooser fileChooser = new FileChooser();
		fileChooser.setTitle("Save tabulated function as...");
		fileChooser.getExtensionFilters().add(new ExtensionFilter("JSON Document (*.json)", "*.json"));
		fileChooser.setInitialDirectory(new File(".\\"));
		File file = fileChooser.showSaveDialog(primaryStage);
		if (file == null)
			return;
		try {
			FunctionGUIApp.tabFDoc.saveFunctionAs(file.getAbsolutePath());
		} catch (Throwable e) {
			showErrorMessage("Failed to save function!");
		}
	}

	@FXML
	private void saveFile(ActionEvent av) {
		if (FunctionGUIApp.tabFDoc.isFileNameAssigned())
			try {
				FunctionGUIApp.tabFDoc.saveFunction();
			} catch (Throwable e) {
				showErrorMessage("Failed to save function!");
			}
		else
			saveAsFile(av);
	}

	@FXML
	private void loadFunction(ActionEvent av) {
		// TODO: load function???
	}

	@FXML
	private void tabulateFunction(ActionEvent av) {
		// TODO: tabulate function???
	}

	@FXML
	private void btNewClick(ActionEvent av) {
		edY.setText(edX.getText());
	}

	@FXML
	private void newDocument(ActionEvent av) {
		showDialog();
	}

	@FXML
	private void addPoint(ActionEvent av) {
		try {
			FunctionGUIApp.tabFDoc
					.addPoint(new FunctionPoint(Double.parseDouble(edX.getText()), Double.parseDouble(edY.getText())));
		} catch (Throwable e) {
			showErrorMessage(e.getLocalizedMessage());
		}
	}

	@FXML
	private void deletePoint(ActionEvent av) {
		try {
			FunctionGUIApp.tabFDoc.deletePoint(FunctionGUIApp.tabFDoc.getPointsCount() - 1);
		} catch (Throwable e) {
			showErrorMessage(e.getLocalizedMessage());
		}
	}

	public void redraw() {
		if (!table.getColumns().isEmpty())
			table.getItems().clear();
		for (int i = 0; i < FunctionGUIApp.tabFDoc.getPointsCount(); ++i) {
			FunctionPointT point = new FunctionPointT(FunctionGUIApp.tabFDoc.getPointX(i),
					FunctionGUIApp.tabFDoc.getPointY(i));
			table.getItems().add(point);
		}
		labelPointNumber.setText("Count of points: " + FunctionGUIApp.tabFDoc.getPointsCount());
	}

	// Render

	@Override
	public void initialize(URL location, ResourceBundle resources) {
		columnX.setCellValueFactory(new PropertyValueFactory<FunctionPointT, Double>("x"));
		table.getColumns().add(columnX);
		columnY.setCellValueFactory(new PropertyValueFactory<FunctionPointT, Double>("y"));
		table.getColumns().add(columnY);
		labelPointNumber.setOnMouseReleased(new EventHandler<MouseEvent>() {
			public void handle(MouseEvent event) {
				labelPointNumber.setText("Count of points: " + FunctionGUIApp.tabFDoc.getPointsCount());
			}
		});
		table.setRowFactory(tableView -> {
			TableRow<FunctionPointT> row = new TableRow<FunctionPointT>();
			row.setOnMouseReleased(new EventHandler<MouseEvent>() {
				public void handle(MouseEvent mouseEvent) {
					if (row.getIndex() < FunctionGUIApp.tabFDoc.getPointsCount())
						labelPointNumber
								.setText(String.format("Point %d of %d", row.getIndex() + 1,
										FunctionGUIApp.tabFDoc.getPointsCount()));
				}
			});
			return row;
		});
		for (int i = 0; i < FunctionGUIApp.tabFDoc.getPointsCount(); ++i) {
			FunctionPointT point = new FunctionPointT(FunctionGUIApp.tabFDoc.getPointX(i),
					FunctionGUIApp.tabFDoc.getPointY(i));
			table.getItems().add(point);
		}
	}

	public int showDialog() {
		if (dialogStage == null) {
			try {
				dialogStage = new Stage();
				FXMLLoader loader = new FXMLLoader(getClass().getResource("FXMLNewDocForm.fxml"));
				Parent root = loader.load();
				dialogController = loader.getController();
				dialogController.setStage(dialogStage);
				Scene scene = new Scene(root);
				dialogStage.setTitle("Function parameters");
				dialogStage.setResizable(false);
				dialogStage.setScene(scene);
				dialogStage.initModality(Modality.APPLICATION_MODAL);
				dialogStage.initOwner(primaryStage);
				dialogStage.showAndWait();
			} catch (Throwable e) {
				e.printStackTrace();
				System.exit(-1);
			}
		} else {
			dialogStage.showAndWait();
		}
		return dialogController.getStatus();
	}

	private void showErrorMessage(String message) {
		Alert errorMessage = new Alert(AlertType.ERROR);
		errorMessage.setHeaderText("Error");
		errorMessage.setContentText(message);
		errorMessage.setResizable(false);
		errorMessage.showAndWait();
	}

}