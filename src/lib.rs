use std::fmt::{Display, Formatter, Result};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean() {
        let slr_model = get_setup_data();
        assert_eq!(slr_model.get_x_average(), 2.5);
        assert_eq!(slr_model.get_y_average(), 6.0);
    }

    #[test]
    fn test_line_of_best_fit() {
        let slr_model = get_setup_data();
        let (y_intercept, slope) = slr_model.get_line_of_best_fit();
        assert_eq!(y_intercept, 1.0);
        assert_eq!(slope, 2.0);
    }

    #[test]
    fn test_r_squared() {
        let slr_model = get_setup_data();
        let r_squared = slr_model.get_r_squared();
        assert_eq!(r_squared, 1.0);
    }

    #[test]
    fn test_update_points() {
        let mut slr_model = get_setup_data();

        let mut points: Vec<Point> = Vec::new();
        for value in 1..5 {
            points.push(Point::new(value as f32, (value * 2) as f32));
        }

        slr_model.update_points(&points);

        let (y_intercept, slope) = slr_model.get_line_of_best_fit();
        let r_squared = slr_model.get_r_squared();

        assert_eq!(y_intercept, 0.0);
        assert_eq!(slope, 2.0);
        assert_eq!(r_squared, 1.0);
        assert_eq!(slr_model.get_x_average(), 2.5);
        assert_eq!(slr_model.get_y_average(), 5.0);
    }

    #[test]
    fn test_add_points() {
        let mut slr_model = get_setup_data();

        let mut points: Vec<Point> = Vec::new();
        points.push(Point::new(5.0, 11.0));

        slr_model.add_points(&points);

        let (y_intercept, slope) = slr_model.get_line_of_best_fit();
        let r_squared = slr_model.get_r_squared();

        assert_eq!(y_intercept, 1.0);
        assert_eq!(slope, 2.0);
        assert_eq!(r_squared, 1.0);
        assert_eq!(slr_model.get_x_average(), 3.0);
        assert_eq!(slr_model.get_y_average(), 7.0);
    }
}

#[allow(dead_code)]
fn get_setup_data() -> SLRModel {
    let mut points: Vec<Point> = Vec::new();
    for value in 1..5 {
        points.push(Point::new(value as f32, ((value * 2) + 1) as f32));
    }
    SLRModel::new(&points)
}

#[derive(Clone, Copy, Debug)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}

impl Point {
    pub fn new(x: f32, y: f32) -> Point {
        Point { x, y }
    }
}

pub struct SLRModel {
    data_points: Vec<Point>,
    x_average: f32,
    y_average: f32,
    slope: f32,
    y_intercept: f32,
    sum_square_errors: f32,
    r_squared: f32,
}

impl SLRModel {
    pub fn new(data_points: &Vec<Point>) -> SLRModel  {
        let x_average = SLRModel::set_x_average(data_points);
        let y_average = SLRModel::set_y_average(data_points);
        let (y_intercept, slope) = SLRModel::calculate_line_of_best_fit(data_points, &x_average, &y_average);
        let sum_square_errors = SLRModel::calculate_sum_of_squared_errors(data_points, &x_average, &y_average);
        let r_squared = SLRModel::calculate_r_squared(data_points, &x_average, &y_average);
        SLRModel {
            data_points: data_points.to_vec(),
            x_average,
            y_average,
            slope,
            y_intercept,
            sum_square_errors,
            r_squared
        }
    }

    pub fn update_points(&mut self, data_points: &Vec<Point>) {
        self.x_average = SLRModel::set_x_average(data_points);
        self.y_average = SLRModel::set_y_average(data_points);
        let (y_intercept, slope) = SLRModel::calculate_line_of_best_fit(data_points, &self.x_average, &self.y_average);
        self.y_intercept = y_intercept;
        self.slope = slope;
        self.sum_square_errors = SLRModel::calculate_sum_of_squared_errors(data_points, &self.x_average, &self.y_average);
        self.r_squared = SLRModel::calculate_r_squared(data_points, &self.x_average, &self.y_average);
        self.data_points = data_points.to_vec();
    }

    pub fn add_points(&mut self, data_points: &Vec<Point>) {
        self.data_points.extend(data_points);
        self.x_average = SLRModel::set_x_average(&self.data_points);
        self.y_average = SLRModel::set_y_average(&self.data_points);
        let (y_intercept, slope) = SLRModel::calculate_line_of_best_fit(&self.data_points, &self.x_average, &self.y_average);
        self.y_intercept = y_intercept;
        self.slope = slope;
        self.sum_square_errors = SLRModel::calculate_sum_of_squared_errors(&self.data_points, &self.x_average, &self.y_average);
        self.r_squared = SLRModel::calculate_r_squared(&self.data_points, &self.x_average, &self.y_average);
    }

    pub fn get_data_points(&self) -> Vec<Point> {
        self.data_points.to_vec()
    }

    fn set_x_average(data_points: &Vec<Point>) -> f32 {
        let mut x_sum: f32 = 0.0;
        for point in data_points {
            x_sum += point.x;
        }
        x_sum / data_points.len() as f32
    }

    pub fn get_x_average(&self) -> f32 {
        self.x_average
    }

    fn set_y_average(data_points: &Vec<Point>) -> f32 {
        let mut y_sum: f32 = 0.0;
        for point in data_points {
            y_sum += point.y;
        }
        y_sum / data_points.len() as f32
    }

    pub fn get_y_average(&self) -> f32 {
        self.y_average
    }

    fn calculate_line_of_best_fit(data_points: &Vec<Point>, x_average: &f32, y_average: &f32) -> (f32, f32) {
        let slope: f32 = SLRModel::calculate_slope(data_points, x_average, y_average);
        let y_intercept: f32 = y_average - slope * x_average;
        (y_intercept, slope)
    }

    pub fn get_line_of_best_fit(&self) -> (f32, f32) {
        (self.y_intercept, self.slope)
    }

    fn calculate_sum_of_squared_errors(data_points: &Vec<Point>, x_average: &f32, y_average: &f32) -> f32 {
        let (y_intercept, slope) = SLRModel::calculate_line_of_best_fit(data_points, x_average, y_average);
    
        let mut sum_square_errors: f32 = 0.0;
        for point in data_points {
            sum_square_errors += (point.y - (y_intercept + slope * point.x)).powi(2);
        }
        sum_square_errors
    }

    pub fn get_sum_of_square_errors(&self) -> f32 {
        self.sum_square_errors
    }

    fn calculate_slope(data_points: &Vec<Point>, x_average: &f32, y_average: &f32) -> f32 {
        let sum_squares_xy = SLRModel::calculate_sum_squares_xy(data_points, x_average, y_average);
        let sum_squares_xx = SLRModel::calculate_sum_squares_xx(data_points, x_average);
        sum_squares_xy / sum_squares_xx
    }
    
    fn calculate_sum_squares_xy(data_points: &Vec<Point>, x_average: &f32, y_average: &f32) -> f32 {
        let mut sum: f32 = 0.0;
        for point in data_points {
            sum += (point.x - x_average) * (point.y - y_average);
        }
        sum
    }
    
    fn calculate_sum_squares_xx(data_points: &Vec<Point>, x_average: &f32) -> f32 {
        let mut sum: f32 = 0.0;
        for point in data_points {
            sum += (point.x - x_average).powi(2);
        }
        sum
    }

    fn calculate_total_sum_of_squares(data_points: &Vec<Point>, y_average: &f32) -> f32 {
        let mut sum: f32 = 0.0;
        for point in data_points {
            sum += (point.y - y_average).powi(2);
        }
        sum
    }

    fn calculate_r_squared(data_points: &Vec<Point>, x_average: &f32, y_average: &f32) -> f32 {
        1.0 - (SLRModel::calculate_sum_of_squared_errors(data_points, x_average, y_average) / SLRModel::calculate_total_sum_of_squares(data_points, y_average))
    }

    pub fn get_r_squared(&self) -> f32 {
        self.r_squared
    }
}

impl Display for SLRModel {
    fn fmt(&self, f: &mut Formatter) -> Result {
        let (y_intercept, slope) = self.get_line_of_best_fit();
        write!(f, "Line of best fit: y = {} + {}x \nR-squared: {}", y_intercept, slope, self.get_r_squared())
    }
}

trait SummaryPrint: Display {
    fn summary(&self) {
        println!("SLR Model");
        println!("---------");
        println!("{}", self.to_string());
    }
}

impl SummaryPrint for SLRModel {}
