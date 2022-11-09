
import functions.FunctionPoint;
import gui.Controller;
import gui.FunctionPointT;
import javafx.event.ActionEvent;
import javafx.event.EventHandler;
import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.fxml.Initializable;
import javafx.scene.Parent;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.Menu;
import javafx.scene.control.MenuBar;
import javafx.scene.control.TableColumn;
import javafx.scene.control.TableRow;
import javafx.scene.control.TableView;
import javafx.scene.control.TextField;
import javafx.scene.control.ToolBar;
import javafx.scene.control.cell.PropertyValueFactory;
import javafx.scene.input.MouseButton;
import javafx.scene.input.MouseEvent;
import javafx.stage.Stage;
import javafx.collections.ObservableList;
import javafx.stage.Modality;
import javafx.scene.*;

import java.net.URL;
import java.util.Comparator;
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
			System.err.println(e.getMessage());
		}
	}

	@FXML
	private void deletePoint(ActionEvent av) {
		try {
			FunctionGUIApp.tabFDoc.deletePoint(FunctionGUIApp.tabFDoc.getPointsCount() - 1);
		} catch (Throwable e) {
			System.err.println(e.getMessage());
		}
	}

	public void redraw() {
		if (!table.getColumns().isEmpty())
			table.getItems().clear();
		for (int i = 0; i < FunctionGUIApp.tabFDoc.getPointsCount(); ++i) {
			FunctionPointT point = new FunctionPointT(FunctionGUIApp.tabFDoc.getPointX(i),
					FunctionGUIApp.tabFDoc.getPointY(i));
			table.getItems().add(point);
			System.out.println("(" + columnX.getCellData(i) + " ;" + columnY.getCellData(i));
		}
		// table.refresh();
		labelPointNumber.setText("Count of points: " + FunctionGUIApp.tabFDoc.getPointsCount());
		// for (FunctionPointT i : table.getItems())
		// System.out.println(i.getX().toString() + " " + i.getY());
	}

	@Override
	public void initialize(URL location, ResourceBundle resources) {
		columnX.setCellValueFactory(new PropertyValueFactory<FunctionPointT, Double>("x"));
		table.getColumns().add(columnX);
		columnY.setCellValueFactory(new PropertyValueFactory<FunctionPointT, Double>("y"));
		table.getColumns().add(columnY);
		// table.setEditable(false);
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
			row.setPrefHeight(10);
			row.setMinHeight(10);
			row.setMaxHeight(10);
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
			} catch (Exception e) {
				System.err.println(e.getClass().getSimpleName() + " : " +
						e.getLocalizedMessage());
			}
		} else {
			dialogStage.showAndWait();
		}
		return dialogController.getStatus();
	}

}