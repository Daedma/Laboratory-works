import functions.FunctionPoint;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.TableColumn;
import javafx.scene.control.TableView;
import javafx.scene.control.TextField;
import javafx.scene.control.ToolBar;
import javafx.scene.control.cell.PropertyValueFactory;
import javafx.collections.ObservableList;
import java.net.URL;
import java.util.ResourceBundle;

public class FXMLMainFormController implements Initializable {
	@FXML
	private TextField edY;

	@FXML
	private TextField edX;

	@FXML
	private Label labelPointNumber;

	@FXML
	private TableView<FunctionPointT> table = new TableView<FunctionPointT>();

	@FXML
	private TableColumn columnX = new TableColumn<>("X values");

	@FXML
	private TableColumn columnY = new TableColumn<>("Y values");

	@FXML
	private Button buttonAddPoint;

	@FXML
	private Button buttonDelete;

	@FXML
	private Label labelTextFieldX;

	@FXML
	private Label labelTextFieldY;

	@FXML
	private ToolBar toolBar;

	@FXML
	private Button buttonFile;

	@FXML
	private Button buttonTabulate;

	@FXML
	private void btNewClick(ActionEvent av) {
		edY.setText(edX.getText());
	}

	@FXML
	private void addPoint(ActionEvent av) {
		try {
			FunctionGUIApp.tabFDoc
					.addPoint(new FunctionPoint(Double.parseDouble(edX.getText()), Double.parseDouble(edY.getText())));
		} catch (Throwable e) {
			System.err.println(e.getMessage());
		}
		redraw();
	}

	@FXML
	private void deletePoint(ActionEvent av) {
		try {
			FunctionGUIApp.tabFDoc.deletePoint(FunctionGUIApp.tabFDoc.getPointsCount() - 1);
		} catch (Throwable e) {
			System.err.println(e.getMessage());
		}
		redraw();
	}

	public void redraw() {
		if (!table.getColumns().isEmpty())
			table.getItems().clear();
		for (int i = 0; i < FunctionGUIApp.tabFDoc.getPointsCount(); ++i) {
			FunctionPointT point = new FunctionPointT(FunctionGUIApp.tabFDoc.getPointX(i),
					FunctionGUIApp.tabFDoc.getPointY(i));
			table.getItems().add(point);
		}
		// for (FunctionPointT i : table.getItems())
		// System.out.println(i.getX().toString() + " " + i.getY());
	}

	@Override
	public void initialize(URL location, ResourceBundle resources) {
		table.getColumns().add(columnX);
		columnX.setCellValueFactory(new PropertyValueFactory<FunctionPointT, Double>("x"));
		table.getColumns().add(columnY);
		columnY.setCellValueFactory(new PropertyValueFactory<FunctionPointT, Double>("y"));
		redraw();
	}
}